"""braindecoding.evaluation 子包：检索指标、词级与序列解码评测及统计检验。"""

from __future__ import annotations

import importlib

_LAZY_SYMBOL_MODULES: dict[str, str] = {
    "apply_multi_seed_control_selection": "word_retrieval_evaluation",
    "build_frequency_vocabulary": "vocabulary_coverage",
    "build_l5_sentence_matched_vocabulary": "vocabulary_coverage",
    "run_evaluation_workflow": "word_retrieval_evaluation",
    "summarize_and_export_multi_seed_sequence_decoding": "sequence_decoding_evaluation",
    "summarize_multi_seed_evaluation": "word_retrieval_evaluation",
}


def __getattr__(name: str):
    if name in _LAZY_SYMBOL_MODULES:
        mod_name = _LAZY_SYMBOL_MODULES[name]
        mod = importlib.import_module(f"braindecoding.evaluation.{mod_name}")
        val = getattr(mod, name)
        globals()[name] = val
        return val
    raise AttributeError(f"module 'braindecoding.evaluation' has no attribute {name!r}")
