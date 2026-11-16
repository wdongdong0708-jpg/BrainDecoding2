"""导出论文主文与附录所需的多研究诊断表、定性词/句示例与跨被试耦合结果（薄层 CLI 入口）。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

_REPO_ROOT = str(Path(__file__).resolve().parents[1])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from braindecoding.evaluation.manuscript_study_exporter import (
    run_manuscript_study_exports,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export manuscript tables, qualitative examples, and diagnostics."
    )
    parser.add_argument(
        "--dataset",
        default="pallier2025",
        choices=("pallier2025", "chineseeeg2_littleprince"),
        help="Target dataset identifier.",
    )
    parser.add_argument("--split", default="test", help="Dataset split (default: test).")
    parser.add_argument(
        "--output-dir",
        default="data/results/exports/manuscript_studies",
        help="Directory for exported JSON/CSV study artifacts.",
    )
    parser.add_argument(
        "--export-pt",
        default=None,
        help="Optional path to neural_vocabulary_scores_<split>.pt.",
    )
    parser.add_argument(
        "--decoding-json",
        default=None,
        help="Optional path to sequence_decoding_<split>.json.",
    )
    parser.add_argument(
        "--eval-summary-json",
        default=None,
        help="Optional path to s3_evaluation_summary_<split>.json.",
    )
    parser.add_argument(
        "--word-eval-json",
        default=None,
        help="Optional path to word-level multi_seed_evaluation_<split>.json.",
    )
    parser.add_argument(
        "--event-table-csv",
        default=None,
        help="Optional path to event_table.csv.",
    )
    parser.add_argument(
        "--repo-root",
        default=None,
        help="Optional repository root override.",
    )
    parser.add_argument(
        "--primary-condition",
        default="S3_clean",
        help="Primary condition key for quantile examples and subject coupling.",
    )
    parser.add_argument(
        "--n-permutations",
        type=int,
        default=200,
        help="Number of permutations for Δ_JS duration-prior alignment test.",
    )
    parser.add_argument(
        "--results-dir",
        default=None,
        help="Optional results root containing dataset experiment directories.",
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="Optional preprocessed dataset cache directory override.",
    )
    parser.add_argument(
        "--eval-subdir",
        default="causal_fast_audit",
        help="Word evaluation subdirectory under each run/seed directory.",
    )
    parser.add_argument(
        "--seeds",
        default="0,1,2",
        help="Comma-separated seed indices for 5-model T5-Large diagnostics.",
    )
    parser.add_argument(
        "--run-five-model-diagnostics",
        action="store_true",
        help="Run forward-only T5-Large 5-model synthetic, JS-prior, and Top-10 diagnostics.",
    )
    parser.add_argument(
        "--five-model-diagnostics-json",
        default=None,
        help="Path to an existing five_model_t5_diagnostics_<split>.json summary to materialize.",
    )
    parser.add_argument(
        "--collect-long-window-results",
        action="store_true",
        help="Collect T5-Large isolated long-window and Local r=0/1/2 comparison rows.",
    )
    parser.add_argument(
        "--force-diagnostics",
        action="store_true",
        help="Recompute 5-model T5-Large diagnostics even if cached JSON exists.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_tuple = tuple(
        int(item.strip())
        for item in str(args.seeds).split(",")
        if item.strip()
    )
    manifest = run_manuscript_study_exports(
        dataset=args.dataset,
        split=args.split,
        output_dir=args.output_dir,
        repo_root=args.repo_root,
        export_pt=args.export_pt,
        decoding_json=args.decoding_json,
        eval_summary_json=args.eval_summary_json,
        word_eval_json=args.word_eval_json,
        event_table_csv=args.event_table_csv,
        primary_condition=args.primary_condition,
        n_permutations=args.n_permutations,
        results_dir=args.results_dir,
        cache_dir=args.cache_dir,
        eval_subdir=args.eval_subdir,
        seeds=seed_tuple,
        run_five_model_diagnostics=bool(args.run_five_model_diagnostics),
        five_model_diagnostics_json=args.five_model_diagnostics_json,
        collect_long_window_results=bool(args.collect_long_window_results),
        force_diagnostics=bool(args.force_diagnostics),
    )
    print(
        json.dumps(
            {
                "status": "completed",
                "output_dir": str(Path(args.output_dir).resolve()),
                "exported_files": manifest.get("exported_files", {}),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
