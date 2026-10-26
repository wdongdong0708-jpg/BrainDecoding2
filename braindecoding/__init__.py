"""BrainDecoding 公共代码包。

采用分层子包结构：
- braindecoding.models: 空间投影、扩张卷积脑编码器与局部窗口修订 Transformer
- braindecoding.data: 数据集构建、事件表契约、信号与文本嵌入缓存
- braindecoding.training: 词级与局部窗口滚动修订训练
- braindecoding.decoding: 神经词表得分导出、KV-Cache 语言模型打分、单遍定稿主干束搜索与状态自适应校准
- braindecoding.evaluation: 检索评测指标、序列解码 6 维语义与门控帕累托评估、留出测试集与配对统计检验
- braindecoding.tasks: 数据集专用适配器
"""
from __future__ import annotations

import importlib
import importlib.abc
import importlib.machinery
import importlib.util
from pathlib import Path
import sys

_PKG_DIR = Path(__file__).resolve().parent
_SUBPACKAGE_MAP: dict[str, str] = {
    "chineseeeg2_dataset": "data",
    "common_mode_subtraction_evaluation": "evaluation",
    "context_baseline_training": "training",
    "context_heldout_evaluation": "evaluation",
    "dataset_builder": "data",
    "derived_cache": "data",
    "environment_preflight": "training",
    "evaluation_figures": "evaluation",
    "evaluation_reporting": "evaluation",
    "event_table": "data",
    "full_vocabulary_evaluation": "evaluation",
    "language_model_scoring": "decoding",
    "long_window_dataset": "data",
    "long_window_word_evaluation": "evaluation",
    "multilingual_embedding_training": "training",
    "neural_score_exporter": "decoding",
    "pallier2025_dataset": "data",
    "pallier_alignment_training": "training",
    "protocol_coverage_audit": "training",
    "protocol_stage_training": "training",
    "retrieval_metrics": "evaluation",
    "revision_transition_exports": "evaluation",
    "rolling_beam_search": "decoding",
    "rolling_revision_training": "training",
    "sensor_coordinates": "data",
    "sequence_decoding_evaluation": "evaluation",
    "siamese_debiased_training": "training",
    "state_hyperparameter_calibrator": "decoding",
    "statistical_significance": "evaluation",
    "subspace_coherence_debiasing_evaluation": "evaluation",
    "synthetic_timing_control": "training",
    "synthetic_timing_dataset": "data",
    "synthetic_timing_recovery": "training",
    "text_embeddings": "data",
    "training_queue_scheduler": "training",
    "training_runtime": "training",
    "vocabulary_coverage": "evaluation",
    "word_encoder_training": "training",
    "word_paired_statistics_evaluation": "evaluation",
    "word_retrieval_evaluation": "evaluation",
    "workflow_runner": "training",
}
_MERGED_MODULE_ALIASES: dict[str, tuple[str, str]] = {
    "beam_search": ("decoding", "rolling_beam_search"),
    "build": ("data", "dataset_builder"),
    "candidate_recall": ("evaluation", "retrieval_metrics"),
    "catalog": ("", "protocols"),
    "chineseeeg2": ("data", "chineseeeg2_dataset"),
    "derived": ("data", "derived_cache"),
    "epoch_recovery": ("training", "training_runtime"),
    "events": ("data", "event_table"),
    "full_vocabulary_context_test_v2": ("evaluation", "full_vocabulary_evaluation"),
    "jayalath_significance": ("evaluation", "statistical_significance"),
    "language_model": ("decoding", "language_model_scoring"),
    "natural_word_lm": ("decoding", "language_model_scoring"),
    "natural_word_lm_prompt": ("decoding", "language_model_scoring"),
    "pallier2025": ("data", "pallier2025_dataset"),
    "pallier_alignment_trial": ("training", "pallier_alignment_training"),
    "pallier_bge_trial": ("training", "multilingual_embedding_training"),
    "pallier_curve_exports": ("evaluation", "revision_transition_exports"),
    "pallier_revision_event_exports": ("evaluation", "revision_transition_exports"),
    "preflight": ("training", "environment_preflight"),
    "protocol_catalog": ("", "protocols"),
    "report_layout": ("", "results"),
    "reporting": ("", "results"),
    "retrieval": ("evaluation", "retrieval_metrics"),
    "rolling": ("training", "rolling_revision_training"),
    "runtime": ("training", "training_runtime"),
    "sensors": ("data", "sensor_coordinates"),
    "sequence_decoding_protocol": ("evaluation", "sequence_decoding_evaluation"),
    "single_word_long_window": ("data", "long_window_dataset"),
    "single_word_long_window_test_v2": ("evaluation", "long_window_word_evaluation"),
    "single_word_long_window_test_v2_results": ("evaluation", "long_window_word_evaluation"),
    "statistics": ("evaluation", "statistical_significance"),
    "synthetic_timing": ("data", "synthetic_timing_dataset"),
    "text": ("data", "text_embeddings"),
    "unified_context_baselines_v2": ("training", "context_baseline_training"),
    "unified_context_baselines_v2_report": ("training", "context_baseline_training"),
    "unified_context_heldout_test_v2": ("evaluation", "context_heldout_evaluation"),
    "unified_protocol_audit": ("training", "protocol_coverage_audit"),
    "unified_rolling_training": ("training", "rolling_revision_training"),
    "unified_rolling_v2_queue": ("training", "training_queue_scheduler"),
    "unified_rolling_v2_training": ("training", "protocol_stage_training"),
    "unified_sentence": ("decoding", "rolling_beam_search"),
    "unified_sentence_batch": ("decoding", "rolling_beam_search"),
    "vocabulary": ("evaluation", "vocabulary_coverage"),
    "word": ("training", "word_encoder_training"),
    "word_paired_statistics_test_v2": ("evaluation", "word_paired_statistics_evaluation"),
}


class _SubpackageAliasLoader(importlib.machinery.SourceFileLoader):
    """Loads moved/merged modules and binds all alias keys in sys.modules to the same module object."""

    def __init__(self, fullname: str, path: str, alias_keys: tuple[str, ...], attr_names: tuple[str, ...], subpkg: str) -> None:
        super().__init__(fullname, path)
        self.alias_keys = alias_keys
        self.attr_names = attr_names
        self.subpkg = subpkg

    def create_module(self, spec):
        for key in self.alias_keys:
            existing = sys.modules.get(key)
            if existing is not None:
                return existing
        return None

    def exec_module(self, module) -> None:
        if getattr(module, "__braindecoding_executed__", False):
            for key in self.alias_keys:
                sys.modules[key] = module
            return
        module.__braindecoding_executed__ = True
        for key in self.alias_keys:
            sys.modules[key] = module
        super().exec_module(module)
        for key in self.alias_keys:
            sys.modules[key] = module
        pkg = sys.modules.get("braindecoding")
        subpkg_mod = sys.modules.get(f"braindecoding.{self.subpkg}") if self.subpkg else None
        for attr in self.attr_names:
            if pkg is not None:
                setattr(pkg, attr, module)
            if subpkg_mod is not None:
                setattr(subpkg_mod, attr, module)


class _SubpackageAliasFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname: str, path=None, target=None):
        if not fullname.startswith("braindecoding."):
            return None
        parts = fullname.split(".")
        if len(parts) == 2:
            short = parts[1]
            if short in _SUBPACKAGE_MAP:
                subpkg = _SUBPACKAGE_MAP[short]
                target_stem = short
            elif short in _MERGED_MODULE_ALIASES:
                subpkg, target_stem = _MERGED_MODULE_ALIASES[short]
            elif short in ("protocols", "results"):
                subpkg = ""
                target_stem = short
            else:
                return None
        elif len(parts) == 3 and parts[1] in ("data", "training", "decoding", "evaluation"):
            short = parts[2]
            if short in _SUBPACKAGE_MAP:
                subpkg = _SUBPACKAGE_MAP[short]
                target_stem = short
            elif short in _MERGED_MODULE_ALIASES:
                subpkg, target_stem = _MERGED_MODULE_ALIASES[short]
            else:
                return None
        else:
            return None

        file_path = (_PKG_DIR / subpkg / f"{target_stem}.py") if subpkg else (_PKG_DIR / f"{target_stem}.py")
        if not file_path.is_file():
            return None

        alias_stems = [target_stem] + [
            alias for alias, (s, t) in _MERGED_MODULE_ALIASES.items() if s == subpkg and t == target_stem
        ]
        alias_keys = []
        for stem in alias_stems:
            for sp in ("data", "training", "decoding", "evaluation"):
                if sp == subpkg or stem in _MERGED_MODULE_ALIASES:
                    alias_keys.append(f"braindecoding.{sp}.{stem}")
            alias_keys.append(f"braindecoding.{stem}")

        loader = _SubpackageAliasLoader(fullname, str(file_path), tuple(dict.fromkeys(alias_keys)), tuple(alias_stems), subpkg)
        return importlib.util.spec_from_file_location(fullname, file_path, loader=loader)


if not any(isinstance(f, _SubpackageAliasFinder) for f in sys.meta_path):
    sys.meta_path.insert(0, _SubpackageAliasFinder())


def __getattr__(name: str):
    if name in _SUBPACKAGE_MAP:
        subpkg = _SUBPACKAGE_MAP[name]
        mod = importlib.import_module(f"braindecoding.{subpkg}.{name}")
        sys.modules[f"braindecoding.{name}"] = mod
        globals()[name] = mod
        return mod
    if name in _MERGED_MODULE_ALIASES:
        subpkg, target_stem = _MERGED_MODULE_ALIASES[name]
        canonical = f"braindecoding.{subpkg}.{target_stem}" if subpkg else f"braindecoding.{target_stem}"
        mod = importlib.import_module(canonical)
        sys.modules[f"braindecoding.{name}"] = mod
        if subpkg:
            sys.modules[f"braindecoding.{subpkg}.{name}"] = mod
        globals()[name] = mod
        return mod
    raise AttributeError(f"module 'braindecoding' has no attribute {name!r}")
