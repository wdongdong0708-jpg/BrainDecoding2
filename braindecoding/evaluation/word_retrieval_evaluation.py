"""词级检索评估、负对照解析、多种子汇总与跨模型配对比较的统一公共门面。

各子模块按职责清晰解耦：
- `word_retrieval_statistics.py`：scipy.stats Student-t / McNemar / 配对块检验；
- `word_retrieval_metrics.py`：全词表与多尺度检索指标计算；
- `word_retrieval_controls.py`：时间错位与零信号负对照解析；
- `word_retrieval_common_horizon.py`：common_horizon_delay 等事件子集过滤与 .pt 复评；
- `word_retrieval_single_stage.py`：单阶段留出集评估主执行器；
- `word_retrieval_control_selection.py`：负对照指标注册表与方法投影；
- `word_retrieval_multiseed_controls.py`：跨种子负对照聚合；
- `word_retrieval_multiseed.py`：多种子评估汇总与 JSON 导出；
- `word_retrieval_cross_model.py`：跨模型配对统计比较；
- `word_retrieval_runner.py`：单/全阶段与多种子评估工作流编排。
"""

from __future__ import annotations

from braindecoding.evaluation.word_retrieval_common_horizon import (
    _apply_eval_subdir_to_seed_dir,
    _build_common_horizon_event_filter,
    _evaluate_common_horizon_subset_from_pt,
    _is_encoded_cache_compatible,
    _resolve_eval_subdir,
    _slice_encoded_dict,
)
from braindecoding.evaluation.word_retrieval_control_selection import (
    _CONTROL_METRIC_ALIASES,
    _CONTROL_METRIC_REGISTRY,
    _enrich_method_with_selected_metrics,
    _extract_clean_and_shifted_metric,
    _extract_metric_from_control_agg,
    _parse_control_methods_list,
    _resolve_control_metric_spec,
    _resolve_control_metric_specs,
    _select_active_control_method,
    _select_control_methods_list,
    apply_multi_seed_control_selection,
)
from braindecoding.evaluation.word_retrieval_controls import (
    _extract_control_summary_row,
    _format_shift_label,
    _parse_shift_seconds_list,
    _resolve_control_specification,
    _summarize_shifted_conditions,
    extract_control_summary_row,
    resolve_control_specification,
    summarize_shifted_conditions,
)
from braindecoding.evaluation.word_retrieval_cross_model import (
    CROSS_MODEL_CANONICAL_SLOTS,
    CROSS_MODEL_METRIC_SPECS,
    _canonicalize_cross_model_slot_key,
    _compute_two_model_paired_metric_test,
    _extract_cross_model_slot_record,
    _is_debiased_summary,
    build_cross_model_paired_comparison,
)
from braindecoding.evaluation.word_retrieval_metrics import (
    _build_enriched_full_vocabulary_metrics,
    _compute_encoded_evaluation_metrics,
    _compute_full_vocabulary_horizon_details,
    _score_full_vocabulary_population,
    build_enriched_full_vocabulary_metrics,
    compute_encoded_evaluation_metrics,
    compute_full_vocabulary_horizon_details,
    score_full_vocabulary_population,
)
from braindecoding.evaluation.word_retrieval_multiseed import (
    _aggregate_single_stage_multi_seed,
    _infer_split_from_seed_results,
    _load_seed_exploratory_json,
    _repair_control_entry_clean_ref,
    _repair_legacy_seed_evaluation_payload,
    summarize_multi_seed_evaluation,
)
from braindecoding.evaluation.word_retrieval_multiseed_controls import (
    _aggregate_control_entries_across_seeds,
)
from braindecoding.evaluation.word_retrieval_runner import (
    _parse_seed_list,
    run_evaluation_workflow,
)
from braindecoding.evaluation.word_retrieval_single_stage import (
    _evaluate_single_stage,
    evaluate_single_stage,
)
from braindecoding.evaluation.word_retrieval_statistics import (
    DEFAULT_CAUSAL_CMS_EMA_DECAY,
    DEFAULT_CAUSAL_CMS_ETAS,
    DEFAULT_CONTROL_SHIFTS_SECONDS,
    DEFAULT_POS_CAUSAL_CMS_ETAS,
    DEFAULT_POS_CAUSAL_EMA_DECAY,
    DEFAULT_POS_MIN_BUCKET_COUNT,
    DEFAULT_SUBTRACTION_ETAS,
    STAGE_DISPLAY_NAMES,
    _CachedEvalDatasetProxy,
    _compute_mcnemar_statistics,
    _compute_paired_block_significance,
    _compute_transition_counts,
    _compute_two_class_linguistic_pvalue,
    _format_p_value_with_stars,
    _one_sample_t_test_pvalue,
    _significance_stars,
    _student_t_two_sided_pvalue,
    _theoretical_chance_for_metric,
    normalize_cms_mode,
    parse_eta_values,
)

__all__ = [
    "CROSS_MODEL_CANONICAL_SLOTS",
    "CROSS_MODEL_METRIC_SPECS",
    "DEFAULT_CONTROL_SHIFTS_SECONDS",
    "_aggregate_control_entries_across_seeds",
    "_aggregate_single_stage_multi_seed",
    "_build_common_horizon_event_filter",
    "_build_enriched_full_vocabulary_metrics",
    "_compute_encoded_evaluation_metrics",
    "_compute_full_vocabulary_horizon_details",
    "_compute_paired_block_significance",
    "_compute_transition_counts",
    "_evaluate_common_horizon_subset_from_pt",
    "_evaluate_single_stage",
    "_extract_control_summary_row",
    "_format_shift_label",
    "_is_encoded_cache_compatible",
    "_parse_control_methods_list",
    "_parse_seed_list",
    "_parse_shift_seconds_list",
    "_resolve_control_metric_spec",
    "_resolve_control_metric_specs",
    "_resolve_control_specification",
    "_score_full_vocabulary_population",
    "_select_control_methods_list",
    "_significance_stars",
    "_summarize_shifted_conditions",
    "apply_multi_seed_control_selection",
    "build_cross_model_paired_comparison",
    "build_enriched_full_vocabulary_metrics",
    "compute_encoded_evaluation_metrics",
    "compute_full_vocabulary_horizon_details",
    "evaluate_single_stage",
    "extract_control_summary_row",
    "normalize_cms_mode",
    "parse_eta_values",
    "resolve_control_specification",
    "run_evaluation_workflow",
    "score_full_vocabulary_population",
    "summarize_multi_seed_evaluation",
    "summarize_shifted_conditions",
]
