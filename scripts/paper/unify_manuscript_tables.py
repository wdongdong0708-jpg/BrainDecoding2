"""Apply the shared native table style using verified frozen manuscript sources.

No model training, decoding, checkpoint selection or resampling is performed.
Historical image tables are retained on disk but replaced in the manuscript by
native LaTeX tables. All existing numerical result cells must remain unchanged.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import yaml

from scripts.paper import consolidate_results_tables as complete
from scripts.paper import control_results_tables as controls
from scripts.paper import polish_main_assets as presentation
from scripts.paper import simplify_appendix_tables as appendix

TABLES = ROOT / 'paper/tables'
EXPORT = ROOT / 'data/results/exports'
AUDIT = ROOT / 'data/results/audits/terminology_table_style_20261011'
BACKUP = ROOT / '.cache/terminology_table_style_20261011'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def export_protocol_tables():
    """Make the already audited protocol-table cells a reports export."""
    old_path = ROOT / 'data/results/audits/handwritten_revision_20261004/assets.json'
    old = read(old_path)
    sources = {p.replace('\\', '/'): h for p, h in old['source_hashes'].items()}
    for rel, digest in sources.items():
        assert sha(ROOT / rel) == digest, rel
    sources[old_path.relative_to(ROOT).as_posix()] = sha(old_path)
    manifest = read(ROOT / 'data/manifests/rolling_sentence/heldout_test_v2.json')
    config = yaml.safe_load((ROOT / 'configs/training/rolling_revision_protocol.yaml').read_text(encoding='utf-8'))
    decoder, language = manifest['decoder'], config['language_model']
    for key, source in [('candidates', 'candidate_count'), ('beam_width', 'beam_width'),
                        ('neural_scale', 'neural_scale'), ('proposal_count', 'proposal_count'),
                        ('proposal_max_new_tokens', 'proposal_max_new_tokens')]:
        assert decoder[key] == language[source]
    assert config['model']['local_window_revision']['history_radius'] == 2
    assert config['model']['local_window_revision']['horizons'] == [0, 1, 2]
    assert config['training']['revision_cutoff_policy'] == 'all'
    weights = {v['selected_weight'] for v in manifest['datasets'].values()}
    assert len(weights) == 1
    setting_values = [language['model_name'].split('/')[-1] + ' 基础版',
                      str(decoder['candidates']), f"{decoder['neural_scale']:g}",
                      str(decoder['beam_width']), str(decoder['proposal_count']),
                      str(decoder['proposal_max_new_tokens']), str(decoder['revision_words']),
                      str(language['context_tokens']) + ' token',
                      ', '.join(f'{v:g}' for v in language['language_weight_grid']),
                      f'{next(iter(weights)):g}']
    original = old['facts']['protocol_tables']
    assert [r[1] for r in original['sentence_decoder_settings']] == setting_values
    replacements = {'完整片段': '完整上下文', '独立延迟': '独立可见状态',
                    '各延迟分别训练': '各状态分别训练', '滚动回修': '共享滚动模型'}
    model_rows = [[replacements.get(cell, cell) for cell in row]
                  for row in original['model_settings']]
    decoder_rows = [[row[0], value]
                    for row, value in zip(original['sentence_decoder_settings'], setting_values)]
    decoder_rows[6][0] = '近期重搜词位数（B/C）'
    record = {'status': 'verified_frozen_settings', 'sources': sources,
              'model_settings': model_rows, 'sentence_decoder_settings': decoder_rows,
              'new_training_decoding_or_resampling': False}
    path = EXPORT / 'manuscript_protocol_tables_v2.json'
    if path.exists():
        assert read(path) == record, 'Frozen protocol-table export changed.'
    else:
        path.write_text(json.dumps(record, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    # Native tables consume the canonical reports export rather than paper data.
    return read(path), {**sources, path.relative_to(ROOT).as_posix(): sha(path)}


def protocol_tables(record):
    model_rows = [list(row) for row in record['model_settings']]
    model_rows[-1][2] = r'\shortstack[l]{遍历到达时刻\\监督 K0/K1/K2}'
    model = appendix.table('llll', ['模型', '输入范围', '训练目标', '参数关系'], model_rows)
    decoder = appendix.table('ll', ['设置', '两个数据集共用的取值'], record['sentence_decoder_settings'])
    return {'model_settings_v2.tex': model, 'sentence_decoder_settings_v2.tex': decoder}


def rolling_table(paired):
    rows, facts = [], []
    for dataset, label in presentation.DATASETS:
        for horizon in ['K0', 'K1', 'K2']:
            part = paired[(paired.dataset == dataset) & (paired.horizon == horizon)]
            assert len(part) == 3 and set(part.seed) == {0, 1, 2}
            cells = [label if horizon == 'K0' else '', horizon]
            for model in ['independent', 'rolling']:
                values = 100 * part[model + '_macro_recall_at_10']
                cells.append('$' + f'{values.mean():.2f}\\pm{values.std(ddof=1):.2f}' + '$')
            effect = 100 * part['rolling_minus_independent_macro_recall_at_10'].mean()
            cells.append('$' + f'{effect:+.2f}' + '$')
            rows.append(cells)
            facts.append({'dataset': dataset, 'state': horizon,
                          'independent': cells[2], 'shared': cells[3], 'difference': cells[4]})
    table = appendix.table('llrrr', ['数据集', '可见状态', '独立模型', '共享滚动模型', '差值 (pp)'],
                           rows, break_before=(3,))
    return table, facts


def data_cells(text):
    """Compare numerical body cells independent of typography/header units."""
    data = []
    in_body = False
    for line in text.splitlines():
        if r'\midrule' in line:
            in_body = True
            continue
        if any(token in line for token in [r'\endfirsthead', r'\endhead', r'\endfoot']):
            in_body = False
        if r'\endlastfoot' in line:
            in_body = True
            continue
        if not in_body or '&' not in line or r'\\' not in line:
            continue
        # Longtable continued headers are repeated presentation, not result cells.
        if '数据集' in line or '输入/模型' in line or r'\multicolumn' in line:
            continue
        numeric = []
        for cell in line.split('&'):
            if re.search(r'\d+\.\d+', cell) and not re.search(r'[A-Za-z\u3400-\u9fff]',
                    re.sub(r'\\(?:textbf|pm|mathbf)\b', '', cell)):
                numeric.append(re.findall(r'[+-]?\d+\.\d+', cell))
            elif re.fullmatch(r'\s*\d+\s*(?:\\\\)?\s*', cell):
                numeric.append([cell.strip().removesuffix(r'\\').strip()])
        if numeric:
            data.append(numeric)
    return data


def main():
    before = read(BACKUP / 'before.json')
    AUDIT.mkdir(parents=True, exist_ok=True)
    context, paired, sentence, core, _, means, _, comparisons, _, vocab, sources = presentation.inputs()
    long, long_sources = controls.verified_study_result('single_word_long_window_test_v2')
    prompts, prompt_sources = controls.verified_study_result('pure_language_prompt_sensitivity_test_v2')
    protocol, protocol_sources = export_protocol_tables()
    native_core, core_facts = presentation.core_table(core)
    word, word_facts = complete.complete_word_table(context)
    main_long, main_long_facts = controls.long_window_table(long)
    appendix_long, appendix_long_facts = controls.long_window_complete_table(long)
    language, language_facts = controls.pure_language_table(prompts)
    per_seed, seed_facts = controls.sequence_seed_table()
    rolling, rolling_facts = rolling_table(paired)
    tables = {
        'word_core_paired_effects_v2_test.tex': native_core,
        'full_vocab_comparison_v2_test.tex': presentation.vocabulary_table(vocab),
        'long_window_comparison_v2_test.tex': main_long,
        'sentence_paired_effects_v2_test.tex': presentation.paired_table(sentence, comparisons),
        'sentence_wait_budget_metrics_v2_test.tex': presentation.waiting_table(means),
        'word_complete_metrics_two_datasets_v2_test.tex': word,
        'long_window_complete_metrics_v2_test.tex': appendix_long,
        'pure_language_prompt_metrics_v2_test.tex': language,
        'sentence_by_seed_acf_v2_test.tex': per_seed,
        'rolling_comparison_v2_test.tex': rolling,
        **protocol_tables(protocol),
    }
    for name, content in tables.items():
        (TABLES / name).write_text(content.replace(r'\addlinespace[3pt]', r'\addlinespace'), encoding='utf-8')
    appendix.AUDIT = AUDIT / 'appendix_sources'
    appendix.main()
    appendix_record = read(appendix.AUDIT / 'tables.json')
    sources.update({**long_sources, **prompt_sources, **protocol_sources, **appendix_record['sources']})
    strategy_path = TABLES / 'decoding_strategies.tex'
    strategy = strategy_path.read_text(encoding='utf-8')
    strategy = re.sub(r'\\normalsize\n\\setlength\{\\tabcolsep\}\{5pt\}\n'
                      r'\\renewcommand\{\\arraystretch\}\{1\.2\}', lambda _: r'\papertablestyle', strategy)
    strategy = strategy.replace('C  & 两次回修', 'C  & 滚动修正')
    strategy_path.write_text(strategy, encoding='utf-8')
    numeric_preservation = {}
    original_image_tables = {'model_settings_v2.tex', 'sentence_decoder_settings_v2.tex',
                             'rolling_comparison_v2_test.tex'}
    for rel in before['tables']:
        path = ROOT / rel
        original = (BACKUP / 'before' / rel).read_text(encoding='utf-8')
        current = path.read_text(encoding='utf-8')
        assert r'\papertablestyle' in current, rel
        assert r'\includegraphics' not in current, rel
        assert not re.search(r'\\(?:small|normalsize|setlength\{\\tabcolsep|renewcommand\{\\arraystretch)', current), rel
        if path.name not in original_image_tables:
            assert data_cells(original) == data_cells(current), (rel, data_cells(original), data_cells(current))
            numeric_preservation[rel] = True
    for rel, digest in before['protected'].items():
        assert sha(ROOT / rel) == digest, rel
    record = {'status': 'verified_presentation_only', 'sources': sources,
              'native_tables': len(before['tables']), 'image_tables_converted': sorted(original_image_tables),
              'existing_numeric_body_cells_preserved': numeric_preservation,
              'style': {'font_pt': 10, 'row_stretch': 1.22, 'column_padding_pt': 3.5,
                        'percentage_and_effect_decimals': 2, 'input_wait_decimals': 3,
                        'regular_numerical_weight': True, 'caption_position': 'above'},
              'word_core_rows': core_facts, 'complete_word_rows': word_facts,
              'long_window_budget_rows': main_long_facts, 'long_window_common_rows': appendix_long_facts,
              'language_prompt_rows': language_facts, 'sequence_seed_rows': seed_facts,
              'rolling_comparison_rows': rolling_facts,
              'generated_tables': {rel: sha(ROOT / rel) for rel in before['tables']},
              'protected_assets_and_exports_unchanged': True,
              'new_training_decoding_or_resampling': False}
    (AUDIT / 'tables.json').write_text(json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps({'status': record['status'], 'native_tables': record['native_tables'],
                      'image_tables_converted': len(original_image_tables),
                      'existing_native_tables_preserved': len(numeric_preservation)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
