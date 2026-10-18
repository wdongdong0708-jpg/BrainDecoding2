"""Present completed long-window, prompt-sensitivity and F-local controls.

Study result/audit/manifest identities and upstream file hashes are checked
before publishing canonical reports/exports and native LaTeX tables. No
training, decoding, model selection or resampling is performed.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import pandas as pd

from scripts.paper import polish_main_assets as presentation

EXPORT = ROOT / 'data/results/exports'
TABLES = ROOT / 'paper/tables'
AUDIT = ROOT / 'data/results/audits/control_results_integration_20261011'
DATASETS = presentation.DATASETS
METRICS = ['sentence_bleu1_mean', 'wer', 'aligned_word_accuracy']


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def verified_study_result(stem):
    directory = ROOT / 'data/results/studies' / stem
    result_path, audit_path = directory / 'results.json', directory / 'audit.json'
    result, audit = read(result_path), read(audit_path)
    manifest_path = ROOT / 'data/manifests/rolling_sentence' / f'{stem}.json'
    manifest = read(manifest_path)
    identity = hashlib.sha256(json.dumps(
        {k: v for k, v in manifest.items() if k != 'manifest_sha256'},
        ensure_ascii=False, sort_keys=True).encode()).hexdigest()
    assert result['status'] == 'completed'
    assert audit['status'] in ['passed', 'verified_complete']
    assert identity == manifest['manifest_sha256'] == result['manifest_sha256'] == audit['manifest_sha256']
    assert sha(result_path) == audit.get('results_sha256', audit.get('result_sha256'))
    hashes = {p.relative_to(ROOT).as_posix(): sha(p)
              for p in [result_path, audit_path, manifest_path]}
    for name in set(audit.get('source_sha256', {})) & set(result.get('source_sha256', {})):
        assert audit['source_sha256'][name] == result['source_sha256'][name], name
    upstream = {**audit.get('source_sha256', {}), **result.get('source_sha256', {})}
    for name, expected in upstream.items():
        path = Path(name)
        if not path.is_absolute():
            path = ROOT / path
        assert sha(path) == expected, name
        hashes[path.relative_to(ROOT).as_posix()] = expected
    destination = EXPORT / f'{stem}_summary.json'
    if destination.exists():
        assert sha(destination) == sha(result_path), str(destination)
    else:
        shutil.copyfile(result_path, destination)
    hashes[destination.relative_to(ROOT).as_posix()] = sha(destination)
    return read(destination), hashes


def long_window_table(result):
    lines = presentation.table_start('llrrl', 5, 1.20)
    lines += [r'数据集 & 比较 & 长窗 & 跨词 & 差值 (pp) [$95\%$ CI] \\', r'\midrule']
    facts = []
    for index, (dataset, label) in enumerate(DATASETS):
        if index:
            lines.append(r'\midrule')
        for horizon in [1, 2]:
            rows = [row for row in result['comparisons'] if row['dataset'] == dataset
                    and row['horizon'] == horizon and row['population'] == 'within_long_window_budget']
            assert len(rows) == 1
            row = rows[0]
            metric = row['top10']
            presentation.waiting.close(metric['right'] - metric['left'], metric['effect'])
            window = f'{row["actual_seconds"]:.2f}'
            interval = '$' + f'{100 * metric["effect"]:+.2f}\\;[{100 * metric["ci_lower"]:.2f}, {100 * metric["ci_upper"]:.2f}]' + '$'
            cells = [label if horizon == 1 else '', f'K{horizon} vs.\\ {window} s',
                     f'{100 * metric["left"]:.2f}', f'{100 * metric["right"]:.2f}', interval]
            lines.append(' & '.join(cells) + r' \\')
            facts.append(row)
    return presentation.finish(lines), facts


def long_window_complete_table(result):
    lines = presentation.table_start('lrrrr', 6, 1.20)
    lines += [r'模型 & \multicolumn{2}{c}{\textbf{Pallier2025}} & \multicolumn{2}{c}{\textbf{ChineseEEG-2}} \\',
              r'\cmidrule(lr){2-3}\cmidrule(lr){4-5}',
              r' & Top-$1$ & Top-$10$ & Top-$1$ & Top-$10$ \\', r'\midrule']
    conditions = [('word_1s', '单词 1.00 s'), ('long_1p27', '单词 1.28 s'),
                  ('long_1p5', '单词 1.50 s'), ('rolling_k0', '共享 K0'),
                  ('rolling_k1', '共享 K1'), ('rolling_k2', '共享 K2')]
    facts = []
    for index, (condition, label) in enumerate(conditions):
        if index == 3:
            lines.append(r'\midrule')
        cells = [label]
        for dataset, _ in DATASETS:
            rows = [row for row in result['summaries'] if row['dataset'] == dataset
                    and row['condition'] == condition and row['population'] == 'common_words'
                    and row['candidates'] == 'full_training_vocabulary']
            assert len(rows) == 1
            row = rows[0]
            assert row['seeds'] == 3
            for metric in ['subject_equal_recall_at_1', 'subject_equal_recall_at_10']:
                cells.append('$' + f'{100 * row[metric + "_mean"]:.2f}\\pm{100 * row[metric + "_sd"]:.2f}' + '$')
            facts.append(row)
        lines.append(' & '.join(cells) + r' \\')
    return presentation.finish(lines), facts


def pure_language_table(result):
    assert result['independent_language_replicates_per_prompt'] == 1
    assert result['test_metrics_used_for_prompt_selection'] is False
    assert result['neural_reference_has_same_prompt'] is False
    lines = presentation.table_start('llrrrr', 7, 1.20)
    lines += [r'提示 & 策略 & \multicolumn{2}{c}{\textbf{Pallier2025}} & \multicolumn{2}{c}{\textbf{ChineseEEG-2}} \\',
              r'\cmidrule(lr){3-4}\cmidrule(lr){5-6}',
              r' & & BLEU-1 $\uparrow$ & WER $\downarrow$ & BLEU-1 $\uparrow$ & WER $\downarrow$ \\', r'\midrule']
    facts = []
    for index, (version, label) in enumerate([('original_prompt', '原提示'), ('selected_prompt', '叙述提示')]):
        if index:
            lines.append(r'\midrule')
        for j, condition in enumerate(['LM_A', 'LM_C1', 'LM_R']):
            cells = [label if j == 0 else '', condition.replace('_', r'\_')]
            for dataset, _ in DATASETS:
                rows = [row for row in result['datasets'][dataset]['summary']
                        if row['prompt_version'] == version and row['condition'] == condition]
                assert len(rows) == 1
                row = rows[0]
                cells += [f'{100 * row["sentence_bleu1_mean"]:.2f}', f'{100 * row["wer"]:.2f}']
                facts.append({'dataset': dataset, **row})
            lines.append(' & '.join(cells) + r' \\')
    return presentation.finish(lines), facts


def sequence_seed_table():
    original = pd.read_csv(EXPORT / 'sentence_wait_budget_test_v2_by_seed.csv')
    controlled = pd.read_csv(EXPORT / 'f_local_sentence_test_v2_by_seed.csv')
    lines = presentation.table_start('llrrr', 8, 1.13)
    lines += [r'数据集 & 策略/种子 & BLEU-1 $\uparrow$ & WER $\downarrow$ & Acc. $\uparrow$ \\', r'\midrule']
    facts = []
    for index, (dataset, label) in enumerate(DATASETS):
        if index:
            lines.append(r'\midrule')
        for seed in [0, 1, 2]:
            if seed:
                lines.append(r'\addlinespace')
            for j, condition in enumerate(['A', 'C', 'F-local', 'F']):
                frame = original if condition == 'A' else controlled
                rows = frame[(frame.dataset == dataset) & (frame.seed == seed) & (frame.condition == condition)]
                assert len(rows) == 1
                row = rows.iloc[0]
                if condition in ['C', 'F']:
                    old = original[(original.dataset == dataset) & (original.seed == seed)
                                   & (original.condition == condition)].iloc[0]
                    for metric in METRICS:
                        presentation.waiting.close(row[metric], old[metric])
                cells = [label if seed == 0 and j == 0 else '', f'{condition} / {seed}']
                cells += [f'{100 * row[metric]:.2f}' for metric in METRICS]
                lines.append(' & '.join(cells) + r' \\')
                facts.append({'dataset': dataset, 'seed': seed, 'condition': condition,
                              **{metric: float(row[metric]) for metric in METRICS}})
    return presentation.finish(lines), facts


def main():
    long, long_sources = verified_study_result('single_word_long_window_test_v2')
    prompts, prompt_sources = verified_study_result('pure_language_prompt_sensitivity_test_v2')
    assert long['language_model_used'] is False
    assert long['cross_word_transformer_in_long_models'] is False
    assert prompts['original_pure_language_metrics_reproduced'] is True
    anomaly = ROOT / 'data/results/studies/pure_language_baseline_test_v2_anomaly_audit'
    anomaly_audit = read(anomaly / 'audit.json')
    assert anomaly_audit['status'] == 'verified_complete'
    assert sha(anomaly / 'results.json') == anomaly_audit['result_sha256']
    _, _, sentence, _, _, _, _, comparisons, _, _, base_sources = presentation.inputs()
    main_long, main_facts = long_window_table(long)
    complete_long, complete_facts = long_window_complete_table(long)
    language, language_facts = pure_language_table(prompts)
    per_seed, seed_facts = sequence_seed_table()
    tables = {
        'long_window_comparison_v2_test.tex': main_long,
        'long_window_complete_metrics_v2_test.tex': complete_long,
        'pure_language_prompt_metrics_v2_test.tex': language,
        'sentence_by_seed_acf_v2_test.tex': per_seed,
        'sentence_paired_effects_v2_test.tex': presentation.paired_table(sentence, comparisons),
    }
    for name, contents in tables.items():
        if name != 'sentence_paired_effects_v2_test.tex':
            contents = contents.replace('polish_main_assets_test_v2.py.',
                                        'control_results_tables_test_v2.py.')
        (TABLES / name).write_text(contents, encoding='utf-8')
    sources = {**long_sources, **prompt_sources, **base_sources}
    for name in ['results.json', 'audit.json']:
        path = anomaly / name
        sources[path.relative_to(ROOT).as_posix()] = sha(path)
    provenance = {
        'status': 'verified_presentation_only',
        'sources': sources,
        'canonical_studies': ['single_word_long_window_test_v2', 'pure_language_prompt_sensitivity_test_v2',
                              'f_local_sentence_test_v2'],
        'long_window_budget_rows': main_facts,
        'long_window_common_rows': complete_facts,
        'language_prompt_rows': language_facts,
        'sequence_seed_rows': seed_facts,
        'long_window_metric': 'full training vocabulary; subject-equal Top-10; three-seed mean',
        'long_window_budget': 'matched last input end; historical context and architecture remain different',
        'language_replicates': 1,
        'language_prompt_scope': 'original neural results share original prompt only',
        'paired_table': {'comparisons': 8, 'datasets': 2, 'previous_six_intervals_preserved': True},
        'f_local_boundary_states': 'interior K2; last two positions K1/K0, as in C',
        'equivalence_test': False,
        'new_training_decoding_or_resampling': False,
        'generated_tables': {name: sha(TABLES / name) for name in tables},
    }
    AUDIT.mkdir(parents=True, exist_ok=True)
    (AUDIT / 'tables.json').write_text(json.dumps(provenance, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    (EXPORT / 'control_results_presentation_test_v2_sources.json').write_text(
        json.dumps({'status': provenance['status'], 'sources': sources,
                    'new_training_decoding_or_resampling': False}, ensure_ascii=False, indent=2) + '\n',
        encoding='utf-8')
    print(json.dumps({'status': provenance['status'], 'tables': len(tables),
                      'paired_comparisons': 8, 'seed_rows': len(seed_facts),
                      'prompt_versions': 2}, ensure_ascii=False))


if __name__ == '__main__':
    main()
