"""Polish the five main figures and four tables from frozen v2 exports.

Run without arguments for vector/PNG previews, then use --deploy to replace
the assets referenced by main.tex. No manuscript prose, scientific exports,
training, decoding, checkpoint selection, or resampling is changed.
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
import pandas as pd
import numpy as np

from scripts.paper.publication_suite_style import style
from scripts.paper import apply_publication_style as method
from scripts.paper import refine_publication_style as renderer
from scripts.paper import word_statistics_and_appendix as words
from scripts.paper import wait_budget_figures as waiting
from scripts.paper import add_revision_diagnostic_figures as edits
from scripts.paper import full_vocabulary_comparison_table as full_vocab
from scripts.paper.update_test_only_word_results import verify as verify_word
from scripts.paper.update_heldout_sentence_results import verify as verify_sentence

EXPORT = ROOT / 'data/results/exports'
OUT = ROOT / 'paper/figures/publication_suite_test_v2_polished'
AUDIT = ROOT / 'data/results/audits/publication_polish_20261010'
DATASETS = waiting.DATASETS
FIGURES = {
    'method_overview_editorial': 'paper/figures/rolling_revision_method',
    'word_core_evidence_v2_test':
        'paper/figures/publication_suite_test_v2_word_statistics/word_core_evidence_v2_test',
    'quality_wait_ac1cf_v2_test':
        'paper/figures/publication_suite_test_v2_wait_budget/quality_wait_ac1cf_v2_test',
    '10_sentence_revision_examples': 'paper/figures/sentence_revision_examples_v2_test',
    '13_display_edit_composition': 'paper/figures/sentence_display_edit_composition_v2_test',
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def verify_hashes(values):
    for rel, expected in values.items():
        assert sha(ROOT / rel) == expected, rel


def verified_f_local():
    """Reuse the completed representation/search control without resampling."""
    stem = 'f_local_sentence_test_v2'
    path = EXPORT / f'{stem}_summary.json'
    summary = read(path)
    audit_path = ROOT / 'data/results/studies' / stem / 'audit.json'
    audit = read(audit_path)
    manifest_path = ROOT / 'data/manifests/rolling_sentence' / f'{stem}.json'
    manifest = read(manifest_path)
    identity = hashlib.sha256(json.dumps(
        {k: v for k, v in manifest.items() if k != 'manifest_sha256'},
        ensure_ascii=False, sort_keys=True).encode()).hexdigest()
    assert summary['status'] == 'completed' and summary['split'] == 'test'
    assert audit['status'] == 'verified_complete'
    assert identity == summary['manifest_sha256'] == audit['manifest_sha256']
    assert sha(path) == audit['result_sha256']
    assert summary['original_C_F_metrics_reproduced'] is True
    assert summary['input_wait_F_local_equals_F_per_event'] is True
    assert summary['new_significance_tests'] is False
    verify_hashes(summary['exports'])
    assert summary['exports'] == audit['exports']
    for source in summary['sources']:
        assert sha(ROOT / source['path']) == source['sha256'], source['path']
    hashes = {str(p.relative_to(ROOT).as_posix()): sha(p)
              for p in [path, audit_path, manifest_path]}
    hashes.update({source['path']: source['sha256'] for source in summary['sources']})
    hashes.update(summary['exports'])
    return summary, hashes


def inputs():
    _, paired, context_report = verify_word()
    sentence, _, _ = verify_sentence()
    context = pd.read_csv(EXPORT / 'unified_context_heldout_test_v2_context_by_seed.csv')
    statistics = read(EXPORT / 'word_paired_statistics_test_v2_summary.json')
    assert statistics['status'] == 'verified_complete'
    verify_hashes(statistics['exports'])
    core = pd.read_csv(EXPORT / 'word_paired_statistics_test_v2_summary.csv')
    diagnostics = read(EXPORT / f'{edits.PREFIX}_sources.json')
    assert diagnostics['status'] == 'verified_descriptive_saved_test_diagnostics'
    assert diagnostics['split'] == 'test'
    assert diagnostics['manifest_sha256'] == sentence['manifest_sha256']
    verify_hashes({**diagnostics['sources'], **diagnostics['exports']})
    totals = pd.read_csv(EXPORT / f'{edits.PREFIX}_display_edits_summary.csv')
    provenance = read(EXPORT / f'{waiting.PREFIX}_sources.json')
    assert provenance['status'] == 'verified_complete'
    assert provenance['conditions'] == waiting.CONDITIONS
    verify_hashes({**provenance['sources'], **provenance['exports']})
    means = pd.read_csv(EXPORT / f'{waiting.PREFIX}_summary.csv')
    seeds = pd.read_csv(EXPORT / f'{waiting.PREFIX}_by_seed.csv')
    comparisons = pd.read_csv(EXPORT / f'{waiting.PREFIX}_supplementary_comparisons.csv')
    ablation, ablation_hashes = waiting.verified_study('fixed_k0_rolling_search_test_v2_stable')
    ablation_seeds = pd.read_csv(EXPORT / 'fixed_k0_rolling_search_test_v2_stable_by_seed.csv')
    b_rows = []
    for dataset, _ in DATASETS:
        part = ablation_seeds[(ablation_seeds.dataset == dataset) & (ablation_seeds.condition == 'B')]
        assert set(part.seed) == {0, 1, 2} and len(part) == 3
        row = means[(means.dataset == dataset) & (means.condition == 'C')].iloc[0].to_dict()
        row['condition'] = 'B'
        for metric in ['sentence_bleu1_mean', 'wer', 'bleu1', 'aligned_word_accuracy']:
            row[metric] = float(part[metric].mean())
        b_rows.append(row)
    means = pd.concat([means, pd.DataFrame(b_rows)], ignore_index=True)
    _, f_local_hashes = verified_f_local()
    f_local = pd.read_csv(EXPORT / 'f_local_sentence_test_v2_comparisons.csv')
    selected = f_local[(f_local.metric == 'sentence_bleu1_mean') &
                       f_local.comparison.isin(['F-local_minus_C', 'F_minus_F-local'])].copy()
    assert len(selected) == 4
    selected = selected.rename(columns={'improvement': 'effect'})
    selected['analysis'] = 'representation_search_control'
    comparisons = pd.concat([comparisons, selected], ignore_index=True)
    full_rows = full_vocab.verified_rows()
    sources = {**context_report['exports'], **statistics['exports'],
               **diagnostics['sources'], **diagnostics['exports'],
               **provenance['sources'], **provenance['exports'], **ablation_hashes,
               **f_local_hashes}
    sources['data/results/exports/full_vocabulary_context_test_v2_summary.json'] = sha(
        EXPORT / 'full_vocabulary_context_test_v2_summary.json')
    sources.update(read(EXPORT / 'full_vocabulary_context_test_v2_summary.json')['exports'])
    sources['data/results/exports/word_paired_statistics_test_v2_summary.json'] = sha(
        EXPORT / 'word_paired_statistics_test_v2_summary.json')
    for dataset, _ in DATASETS:
        raw = context[(context.dataset == dataset) & (context.population == 'all_targets')]
        gain = raw[raw.condition == 'local_r2'].macro_recall_at_10.mean() - raw[
            raw.condition == 'full_context'].macro_recall_at_10.mean()
        expected = core[(core.dataset == dataset) & (core.contrast == 'local_r2_minus_full')
                        & (core.metric == 'macro_recall_at_10')].iloc[0]
        waiting.close(gain, expected.effect)
    return context, paired, sentence, core, totals, means, seeds, comparisons, ablation, full_rows, sources


def table_start(columns, spacing=4, stretch=1.20):
    # Retain the optional arguments for older callers; presentation is centralized.
    return [
        '% Presentation only: verified reports/exports; polish_main_assets_test_v2.py.',
        r'\begingroup', r'\papertablestyle',
        rf'\begin{{tabular}}{{@{{}}{columns}@{{}}}}', r'\toprule',
    ]


def finish(lines):
    return '\n'.join(lines + [r'\bottomrule', r'\end{tabular}', r'\endgroup', ''])


def math_number(value, bold=False):
    text = f'{value:+.2f}'
    return '$' + (r'\mathbf{' + text + '}' if bold else text) + '$'


def core_table(core):
    lines = table_start('lrlrl', 4)
    lines += [
        r'词级比较 & \multicolumn{2}{c}{\textbf{Pallier2025}} & \multicolumn{2}{c}{\textbf{ChineseEEG-2}} \\',
        r'\cmidrule(lr){2-3}\cmidrule(lr){4-5}',
        r' & 差值 (pp) & $95\%$ CI (pp) & 差值 (pp) & $95\%$ CI (pp) \\',
        r'\midrule',
    ]
    facts = []
    for contrast, label in [
        ('local_r2_minus_full', r'局部 $k=2$ $-$ 完整上下文'),
        ('rolling_k2_minus_k0', r'共享 K2 $-$ K0'),
    ]:
        cells = [label]
        for dataset, _ in DATASETS:
            row = core[(core.dataset == dataset) & (core.contrast == contrast)
                       & (core.metric == 'macro_recall_at_10')].iloc[0]
            cells += [math_number(100 * row.effect),
                      f'$[{100 * row.ci_lower:.2f}, {100 * row.ci_upper:.2f}]$']
            facts.append({'dataset': dataset, 'contrast': contrast, 'effect': row.effect,
                          'ci_lower': row.ci_lower, 'ci_upper': row.ci_upper})
        lines.append(' & '.join(cells) + r' \\')
    return finish(lines), facts


def vocabulary_table(rows):
    lines = table_start('llrrrl', 3.5)
    lines += [
        r'数据集 & 指标 & 局部 $k=2$ & 完整上下文 & 差值 (pp) & $95\%$ CI (pp) \\',
        r'\midrule',
    ]
    for index, (dataset, label) in enumerate(DATASETS):
        if index:
            lines.append(r'\midrule')
        for j, (metric, display) in enumerate(full_vocab.METRICS):
            if j == 2:
                lines.append(r'\addlinespace')
            row = next(x for x in rows if x['dataset'] == dataset and x['metric'] == metric)
            cells = [label if j == 0 else '', display,
                     f'{100 * float(row["local_r2_mean"]):.2f}',
                     f'{100 * float(row["full_context_mean"]):.2f}',
                     math_number(100 * float(row['effect'])),
                     f'$[{100 * float(row["ci_lower"]):.2f}, {100 * float(row["ci_upper"]):.2f}]$']
            lines.append(' & '.join(cells) + r' \\')
    return finish(lines)


def paired_table(sentence, comparisons):
    lines = table_start('lcc', 8, 1.20)
    lines += [r'比较 & Pallier2025 & ChineseEEG-2 \\', r'\midrule']
    for contrast, label in [
        ('C_minus_A', r'C $-$ A'), ('B_minus_A', r'B $-$ A'),
        ('C_minus_B', r'C $-$ B'), ('C1_minus_A', r'C1 $-$ A'),
        ('C_minus_C1', r'C $-$ C1'),
        ('F-local_minus_C', r'F-local $-$ C'),
        ('F_minus_F-local', r'F $-$ F-local'), ('C_minus_F', r'C $-$ F'),
    ]:
        cells = [label]
        for dataset, _ in DATASETS:
            if contrast == 'C_minus_A':
                row = sentence['datasets'][dataset]['primary_statistics']
            else:
                selected = comparisons[(comparisons.dataset == dataset) & (comparisons.comparison == contrast)]
                assert len(selected) == 1
                row = selected.iloc[0].to_dict()
            cells.append(f'$ {100 * row["effect"]:+.2f}\\;[{100 * row["ci_lower"]:.2f}, {100 * row["ci_upper"]:.2f}] $')
        lines.append(' & '.join(cells) + r' \\')
    return finish(lines)


def waiting_table(means):
    lines = table_start('lrrrrrrrr', 3, 1.30)
    lines += [
        r'策略 & \multicolumn{4}{c}{\textbf{Pallier2025}} & \multicolumn{4}{c}{\textbf{ChineseEEG-2}} \\',
        r'\cmidrule(lr){2-5}\cmidrule(lr){6-9}',
        r' & \papertablehead{BLEU-1 $\uparrow$}{(\%)} & \papertablehead{WER $\downarrow$}{(\%)} & \papertablehead{Acc. $\uparrow$}{(\%)} & \papertablehead{输入等待}{(s)} & \papertablehead{BLEU-1 $\uparrow$}{(\%)} & \papertablehead{WER $\downarrow$}{(\%)} & \papertablehead{Acc. $\uparrow$}{(\%)} & \papertablehead{输入等待}{(s)} \\',
        r'\midrule',
    ]
    for index, (condition, label) in enumerate([
        ('A', 'A 固定输出'), ('B', 'B 固定表示重搜'), ('C1', 'C1 一次回修'),
        ('C', 'C 滚动修正'), ('F', 'F 完整片段'),
    ]):
        if index in [2, 4]:
            lines.append(r'\addlinespace')
        emphasize = condition in ['C1', 'C']
        cells = [r'\textbf{' + label + '}' if emphasize else label]
        for dataset, _ in DATASETS:
            row = means[(means.dataset == dataset) & (means.condition == condition)].iloc[0]
            values = [f'{100 * row[k]:.2f}' for k in [
                'sentence_bleu1_mean', 'wer', 'aligned_word_accuracy']]
            values.append(f'{row.mean_total_input_wait_s:.3f}')
            cells.extend(values)
        lines.append(' & '.join(cells) + r' \\')
    return finish(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--deploy', action='store_true')
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    AUDIT.mkdir(parents=True, exist_ok=True)
    data = inputs()
    context, paired, sentence, core, totals, means, seeds, comparisons, ablation, full_rows, sources = data
    manuscript_hashes = {p.relative_to(ROOT).as_posix(): sha(p)
                         for p in (ROOT / 'paper/manuscript').glob('*.tex')}
    renderer.OUT = OUT
    method.METHOD = OUT
    waiting.OUT = OUT
    edits.OUT = OUT
    with style(), mpl.rc_context(renderer.FONT):
        method.method_figure()
        words.word_figure(context, paired)
        waiting.plot_tradeoff(means, seeds, comparisons)
        cases = renderer.case_excerpts(sentence)
        renderer.example_figure(cases)
        _, composition_facts = edits.composition_figure(totals)
    native_core, core_facts = core_table(core)
    tables = {
        'word_core_paired_effects_v2_test.tex': native_core,
        'full_vocab_comparison_v2_test.tex': vocabulary_table(full_rows),
        'sentence_paired_effects_v2_test.tex': paired_table(sentence, comparisons),
        'sentence_wait_budget_metrics_v2_test.tex': waiting_table(means),
    }
    preview_tables = ROOT / '.cache/publication_polish_20261010/table_previews'
    preview_tables.mkdir(parents=True, exist_ok=True)
    for name, contents in tables.items():
        (preview_tables / name).write_text(contents, encoding='utf-8')
    artifacts = {}
    for name, destination in FIGURES.items():
        for ext in ['pdf', 'svg', 'png']:
            path = OUT / (name + '.' + ext)
            assert path.exists(), path
            artifacts[path.relative_to(ROOT).as_posix()] = sha(path)
            if args.deploy:
                target = ROOT / (destination + '.' + ext)
                shutil.copyfile(path, target)
                artifacts[target.relative_to(ROOT).as_posix()] = sha(target)
    if args.deploy:
        for name, contents in tables.items():
            path = ROOT / 'paper/tables' / name
            path.write_text(contents, encoding='utf-8')
            artifacts[path.relative_to(ROOT).as_posix()] = sha(path)
    verify_hashes(sources)
    verify_hashes(manuscript_hashes)
    record = {
        'status': 'verified_deployed' if args.deploy else 'verified_preview',
        'presentation_only': True, 'source_sha256': sources, 'artifacts': artifacts,
        'generator_sha256': sha(Path(__file__)), 'main_figures': 5, 'main_tables': 4,
        'mean_seed_numbers_and_material_block_intervals_unchanged': True,
        'quality_wait_main_and_detail_use_actual_total_seconds': True,
        'quality_wait_y_limits_equal_across_datasets': True,
        'edit_detail_denominator': 'all_changed_events_not_renormalized_subset',
        'edit_stage_facts': composition_facts, 'word_core_effects': core_facts,
        'case_selection_and_target_tokens_unchanged': True,
        'manuscript_prose_edited_by_generator': False,
        'new_training_decoding_or_resampling': False,
        'layout_checks': renderer.BOUNDS + edits.CHECKS,
        'pdf_page_inspection_performed': False,
    }
    (AUDIT / 'figures_tables.json').write_text(
        json.dumps(record, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'status': record['status'], 'figures': 5, 'tables': 4,
                      'preview_directory': str(OUT),
                      'sources_verified_and_unchanged': True}, ensure_ascii=False))


if __name__ == '__main__':
    main()
