"""braindecoding.evaluation 子包。"""
from __future__ import annotations

import importlib
import sys

_LAZY_SYMBOL_MODULES: dict[str, str] = {
    "EVALUATION_PALETTE": "evaluation_figures",
    "STAGE_DISPLAY_NAMES": "evaluation_figures",
    "STAGE_SHORT_LABELS": "evaluation_figures",
    "apply_multi_seed_control_selection": "word_retrieval_evaluation",
    "evaluation_plot_context": "evaluation_figures",
    "export_common_mode_subtraction_tables_and_markdown": "evaluation_exports",
    "export_multi_seed_tables_and_markdown": "evaluation_exports",
    "export_sequence_decoding_tables_and_markdown": "evaluation_exports",
    "export_stage_comparison_tables_and_markdown": "evaluation_exports",
    "export_subspace_coherence_debiasing_tables_and_markdown": "evaluation_exports",
    "export_word_evaluation_tables_and_markdown": "evaluation_exports",
    "format_common_mode_subtraction_report": "evaluation_reporting",
    "format_multi_seed_evaluation_report": "evaluation_reporting",
    "format_sequence_decoding_executive_summary": "evaluation_reporting",
    "format_sequence_decoding_report": "evaluation_reporting",
    "format_stage_comparison_report": "evaluation_reporting",
    "format_subspace_coherence_debiasing_report": "evaluation_reporting",
    "format_word_evaluation_report": "evaluation_reporting",
    "plot_baseline_and_shifted_comparison": "evaluation_figures",
    "plot_common_mode_subtraction_figures": "evaluation_figures",
    "plot_confidence_gate_pareto_curve": "evaluation_figures",
    "plot_evaluation_figures": "evaluation_figures",
    "plot_multi_subject_observation_curve": "evaluation_figures",
    "plot_negative_control_figures": "evaluation_figures",
    "plot_stage_comparison_figure": "evaluation_figures",
    "plot_stage_control_comparison_figure": "evaluation_figures",
    "plot_subspace_coherence_debiasing_figures": "evaluation_figures",
    "plot_vocabulary_scaling_curve": "evaluation_figures",
    "print_common_mode_subtraction_report": "evaluation_reporting",
    "print_multi_seed_evaluation_report": "evaluation_reporting",
    "print_sequence_decoding_report": "evaluation_reporting",
    "print_stage_comparison_report": "evaluation_reporting",
    "print_subspace_coherence_debiasing_report": "evaluation_reporting",
    "print_word_evaluation_report": "evaluation_reporting",
    "render_saved_evaluation_report": "evaluation_reporting",
    "replot_from_saved_json": "evaluation_figures",
    "run_evaluation_workflow": "word_retrieval_evaluation",
    "summarize_multi_seed_evaluation": "word_retrieval_evaluation",
}


def __getattr__(name: str):
    if name in _LAZY_SYMBOL_MODULES:
        mod_name = _LAZY_SYMBOL_MODULES[name]
        mod = importlib.import_module(f"braindecoding.evaluation.{mod_name}")
        val = getattr(mod, name)
        globals()[name] = val
        return val

    from braindecoding import _MERGED_MODULE_ALIASES, _SUBPACKAGE_MAP
    if name in _SUBPACKAGE_MAP:
        target_sub = _SUBPACKAGE_MAP[name]
        mod = importlib.import_module(f"braindecoding.{target_sub}.{name}")
        globals()[name] = mod
        return mod
    if name in _MERGED_MODULE_ALIASES:
        target_sub, target_stem = _MERGED_MODULE_ALIASES[name]
        canonical = f"braindecoding.{target_sub}.{target_stem}" if target_sub else f"braindecoding.{target_stem}"
        mod = importlib.import_module(canonical)
        sys.modules[f"braindecoding.evaluation.{name}"] = mod
        globals()[name] = mod
        return mod
    raise AttributeError(f"module 'braindecoding.evaluation' has no attribute {name!r}")
