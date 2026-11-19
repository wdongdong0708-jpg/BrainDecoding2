"""词级评估工作流主入口（支持单阶段、全阶段对比与多种子串联编排）。"""

from __future__ import annotations

import gc
import json
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch

from braindecoding.config import (
    load_yaml_with_extends,
    project_path,
    resolve_config_path,
)

from braindecoding.evaluation.word_retrieval_common_horizon import (
    _apply_eval_subdir_to_seed_dir,
    _resolve_eval_subdir,
)
from braindecoding.evaluation.word_retrieval_multiseed import (
    summarize_multi_seed_evaluation,
)
from braindecoding.evaluation.word_retrieval_single_stage import (
    _evaluate_single_stage,
)
from braindecoding.evaluation.word_retrieval_statistics import (
    STAGE_DISPLAY_NAMES,
    _compute_paired_block_significance,
    export_stage_comparison_tables_and_markdown,
    plot_stage_comparison_figure,
    print_stage_comparison_report,
)
from braindecoding.training.training_runtime import save_json

def _parse_seed_list(
    raw_seed: int | str | Iterable[int | str] | None,
) -> list[int]:
    """解析单种子或多种子输入（支持整数、逗号分隔字符串如 '0,1,2' 或序列如 [0, 1, 2]）。"""
    if raw_seed is None:
        return [0]
    if isinstance(raw_seed, bool):
        return [int(raw_seed)]
    if isinstance(raw_seed, int):
        return [int(raw_seed)]
    if isinstance(raw_seed, str):
        cleaned = raw_seed.strip().strip("[]()")
        if not cleaned:
            return [0]
        tokens = [
            tok.strip()
            for tok in cleaned.replace(";", ",").replace(" ", ",").split(",")
            if tok.strip()
        ]
    else:
        tokens = []
        for item in raw_seed:
            if isinstance(item, str):
                tokens.extend(
                    tok.strip()
                    for tok in item.strip("[]()")
                    .replace(";", ",")
                    .replace(" ", ",")
                    .split(",")
                    if tok.strip()
                )
            else:
                tokens.append(item)
    seeds: list[int] = []
    for tok in tokens:
        val = int(tok)
        if val not in seeds:
            seeds.append(val)
    return seeds or [0]

def run_evaluation_workflow(
    dataset: str | None = None,
    *,
    stage: str | None = None,
    seed: int | str | Iterable[int | str] | None = 0,
    split: str = "val",
    subject: str | None = None,
    num_subjects: int | None = None,
    subjects: str | Iterable[str] | None = None,
    eval_batch_size: int | None = None,
    num_workers: int | None = None,
    precision: str | None = None,
    window_seconds: float | None = None,
    window_start_offset_seconds: float | None = None,
    require_cache: bool = False,
    checkpoint_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    eval_subdir: str | None = None,
    exp_tag: str | None = None,
    word_exp_tag: str | None = None,
    config_path: str | Path | None = None,
    control_config_path: str | Path | None = None,
    shift_seconds: str | Iterable[float | int | str] | None = None,
    shift_mode: str | None = None,
    preroll_seconds: float | None = None,
    cms_mode: str | None = None,
    cms_ema_decay: float | None = None,
    zero_signal_control: bool = False,
    subtraction_etas: str | Iterable[float | int | str] | None = None,
    control_method: str | Iterable[str] | None = None,
    control_metric: str | Iterable[str] | None = None,
    common_horizon_delay: int | None = None,
    all_stages: bool = False,
    plot: bool = True,
    show_progress: bool = True,
    max_eval_rows: int | None = None,
    verbose: bool = False,
    _export_outer_multi_seed: bool = True,
) -> dict:
    """运行验证集 (val) 或测试集 (test) 评估与可视化工作流（支持单阶段、全阶段对比、多种子聚合、负对照及探索性留一法共模扣除）。"""
    from braindecoding.training.workflow_runner import (
        WORKFLOW_STAGES,
        _resolve_results_root_and_exp_tag,
        _resolve_word_exp_tag,
        build_workflow_config,
        workflow_output_directory,
    )

    if split not in {"val", "test"}:
        raise ValueError(f"评估 split 只能是 'val' 或 'test'，收到：{split!r}")

    seed_list = _parse_seed_list(seed)
    if len(seed_list) > 1:
        seed_results: list[dict] = []
        for idx, sd_val in enumerate(seed_list):
            print(
                f"\n{'#' * 86}\n"
                f" [Multi-Seed Evaluation {idx + 1}/{len(seed_list)}] Evaluating seed={sd_val} (all seeds: {seed_list})\n"
                f"{'#' * 86}",
                flush=True,
            )
            res_i = run_evaluation_workflow(
                dataset,
                stage=stage,
                seed=sd_val,
                split=split,
                subject=subject,
                num_subjects=num_subjects,
                subjects=subjects,
                eval_batch_size=eval_batch_size,
                num_workers=num_workers,
                precision=precision,
                window_seconds=window_seconds,
                window_start_offset_seconds=window_start_offset_seconds,
                require_cache=require_cache,
                checkpoint_path=None,
                output_dir=None,
                eval_subdir=eval_subdir,
                exp_tag=exp_tag,
                word_exp_tag=word_exp_tag,
                config_path=config_path,
                control_config_path=control_config_path,
                shift_seconds=shift_seconds,
                shift_mode=shift_mode,
                preroll_seconds=preroll_seconds,
                cms_mode=cms_mode,
                cms_ema_decay=cms_ema_decay,
                zero_signal_control=zero_signal_control,
                subtraction_etas=subtraction_etas,
                control_method=control_method,
                control_metric=control_metric,
                common_horizon_delay=common_horizon_delay,
                all_stages=all_stages,
                plot=plot,
                show_progress=show_progress,
                max_eval_rows=max_eval_rows,
                verbose=verbose,
                _export_outer_multi_seed=False,
            )
            seed_results.append(res_i)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        return summarize_multi_seed_evaluation(
            seed_results,
            split=split,
            output_dir=output_dir,
            verbose=verbose,
            print_report=True,
            control_method=control_method,
            control_metric=control_metric,
            common_horizon_delay=common_horizon_delay,
            eval_subdir=eval_subdir,
        )

    seed = int(seed_list[0])
    if not all_stages:
        single_res = _evaluate_single_stage(
            dataset,
            stage=stage,
            seed=seed,
            split=split,
            subject=subject,
            num_subjects=num_subjects,
            subjects=subjects,
            eval_batch_size=eval_batch_size,
            num_workers=num_workers,
            precision=precision,
            window_seconds=window_seconds,
            window_start_offset_seconds=window_start_offset_seconds,
            require_cache=require_cache,
            checkpoint_path=checkpoint_path,
            output_dir=output_dir,
            eval_subdir=eval_subdir,
            exp_tag=exp_tag,
            word_exp_tag=word_exp_tag,
            config_path=config_path,
            control_config_path=control_config_path,
            shift_seconds=shift_seconds,
            shift_mode=shift_mode,
            preroll_seconds=preroll_seconds,
            cms_mode=cms_mode,
            cms_ema_decay=cms_ema_decay,
            zero_signal_control=zero_signal_control,
            subtraction_etas=subtraction_etas,
            plot=plot,
            show_progress=show_progress,
            max_eval_rows=max_eval_rows,
            verbose=verbose,
            print_report=True,
            common_horizon_delay=common_horizon_delay,
        )
        if plot and common_horizon_delay is None:
            ds_name = single_res["dataset"]
            sub_tag = single_res["subset"]["subset_tag"]
            sd = single_res["seed"]
            active_tag = single_res.get("exp_tag", "")
            active_word_tag = single_res.get("word_exp_tag", active_tag)
            discovered: dict[str, dict] = {single_res["stage"]: single_res}
            for other_st in WORKFLOW_STAGES:
                if other_st == single_res["stage"]:
                    continue
                target_tag = (
                    active_word_tag
                    if other_st == "word" and active_word_tag != active_tag
                    else active_tag
                )
                other_dir = workflow_output_directory(
                    ds_name,
                    other_st,
                    sub_tag,
                    sd,
                    mode="train",
                    exp_tag=target_tag,
                    ignore_implicit_tag=(
                        other_st == "word" and target_tag == ""
                    ),
                )
                isolated_other_file = (
                    _apply_eval_subdir_to_seed_dir(
                        other_dir, eval_subdir, stage=other_st
                    )
                    / f"evaluation_{split}.json"
                )
                other_eval_file = (
                    isolated_other_file
                    if isolated_other_file.is_file()
                    else (other_dir / f"evaluation_{split}.json")
                )
                if not other_eval_file.is_file():
                    smoke_other = (
                        workflow_output_directory(
                            ds_name,
                            other_st,
                            sub_tag,
                            sd,
                            mode="smoke",
                            exp_tag=target_tag,
                            ignore_implicit_tag=(
                                other_st == "word" and target_tag == ""
                            ),
                        )
                        / f"evaluation_{split}.json"
                    )
                    if smoke_other.is_file():
                        other_eval_file = smoke_other
                if (
                    not other_eval_file.is_file()
                    and target_tag
                    and other_st == "word"
                ):
                    fallback_other = (
                        workflow_output_directory(
                            ds_name,
                            other_st,
                            sub_tag,
                            sd,
                            mode="train",
                            ignore_implicit_tag=True,
                        )
                        / f"evaluation_{split}.json"
                    )
                    if fallback_other.is_file():
                        other_eval_file = fallback_other
                if other_eval_file.is_file():
                    try:
                        other_loaded = json.loads(
                            other_eval_file.read_text(encoding="utf-8")
                        )
                        single_nc_conds = (
                            (single_res.get("negative_controls") or {}).get(
                                "conditions"
                            )
                            or []
                        )
                        other_nc_conds = (
                            (other_loaded.get("negative_controls") or {}).get(
                                "conditions"
                            )
                            or []
                        )
                        if single_nc_conds and other_nc_conds:
                            single_keys = {c.get("key") for c in single_nc_conds}
                            other_keys = {c.get("key") for c in other_nc_conds}
                            if single_keys != other_keys:
                                other_loaded = dict(other_loaded)
                                other_loaded["negative_controls"] = {}
                        discovered[other_st] = other_loaded
                    except Exception:
                        pass
            if len(discovered) >= 1:
                comp_figs = plot_stage_comparison_figure(
                    discovered,
                    Path(single_res["output_dir"]) / "figures",
                    split=split,
                )
                single_res["figures"].update(comp_figs)
                save_json(Path(single_res["evaluation_json"]), single_res)
        if _export_outer_multi_seed:
            outer_dir = (
                Path(str(single_res["output_dir"])).resolve().parent
                if output_dir is None
                else Path(output_dir).resolve().parent
            )
            outer_summary = summarize_multi_seed_evaluation(
                [single_res],
                split=split,
                output_dir=outer_dir,
                verbose=verbose,
                print_report=True,
                control_method=control_method,
                control_metric=control_metric,
                common_horizon_delay=common_horizon_delay,
                eval_subdir=eval_subdir,
            )
            single_res["outer_multi_seed_evaluation_json"] = str(
                outer_summary.get("evaluation_json", "")
            )
            single_res["outer_multi_seed_exports"] = outer_summary.get(
                "exports", {}
            )
        return single_res

    base_cfg, subset_spec = build_workflow_config(
        dataset,
        stage="word",
        seed=seed,
        subject=subject,
        num_subjects=num_subjects,
        subjects=subjects,
        window_seconds=window_seconds,
        window_start_offset_seconds=window_start_offset_seconds,
        exp_tag=exp_tag,
        word_exp_tag=word_exp_tag,
        config_path=config_path,
    )
    effective_control_config_path = control_config_path
    if effective_control_config_path is None and config_path is not None:
        resolved_cfg_p = resolve_config_path(project_path(config_path))
        if resolved_cfg_p.is_file():
            raw_explicit = load_yaml_with_extends(resolved_cfg_p)
            if isinstance(raw_explicit, dict) and (
                "control" in raw_explicit
                or "shift_seconds" in (raw_explicit.get("evaluation") or {})
                or bool(
                    (raw_explicit.get("evaluation") or {}).get(
                        "zero_signal_control"
                    )
                )
            ):
                effective_control_config_path = config_path

    ds_name = subset_spec["dataset"]
    sub_tag = subset_spec["subset_tag"]
    sd = int(base_cfg["training"]["seed"])
    base_root, resolved_exp_tag = _resolve_results_root_and_exp_tag(
        base_cfg["workflow"].get("exp_tag", exp_tag)
    )
    has_explicit_word_tag, resolved_word_exp_tag = _resolve_word_exp_tag(
        word_exp_tag
    )

    has_stage2_under_exp_tag = False
    for st2 in ("context", "revision", "prefix_revision"):
        for mode_cand in ("train", "smoke"):
            cand_dir2 = workflow_output_directory(
                ds_name,
                st2,
                sub_tag,
                sd,
                mode=mode_cand,
                exp_tag=resolved_exp_tag,
                ignore_implicit_tag=False,
            )
            if (cand_dir2 / "best.ckpt").is_file() or (
                cand_dir2 / "best.pt"
            ).is_file():
                has_stage2_under_exp_tag = True
                break
        if has_stage2_under_exp_tag:
            break

    available_stages: list[tuple[str, str, bool]] = []
    for st in WORKFLOW_STAGES:
        candidate_tags: list[tuple[str, bool]] = []
        if st == "word":
            if has_explicit_word_tag and (
                has_stage2_under_exp_tag or word_exp_tag is not None
            ):
                candidate_tags.append(
                    (resolved_word_exp_tag, resolved_word_exp_tag == "")
                )
            if (resolved_exp_tag, False) not in candidate_tags:
                candidate_tags.append((resolved_exp_tag, False))
            if (
                has_explicit_word_tag
                and (resolved_word_exp_tag, resolved_word_exp_tag == "")
                not in candidate_tags
            ):
                candidate_tags.append(
                    (resolved_word_exp_tag, resolved_word_exp_tag == "")
                )
            if resolved_exp_tag and ("", True) not in candidate_tags:
                candidate_tags.append(("", True))
        else:
            candidate_tags.append((resolved_exp_tag, False))

        found = False
        for tag_cand, ignore_imp in candidate_tags:
            for mode_candidate in ("train", "smoke"):
                cand_dir = workflow_output_directory(
                    ds_name,
                    st,
                    sub_tag,
                    sd,
                    mode=mode_candidate,
                    exp_tag=tag_cand,
                    ignore_implicit_tag=ignore_imp,
                )
                if (cand_dir / "best.ckpt").is_file() or (
                    cand_dir / "best.pt"
                ).is_file():
                    available_stages.append((st, tag_cand, ignore_imp))
                    found = True
                    break
            if found:
                break

    if not available_stages:
        expected_dir = workflow_output_directory(
            ds_name, "word", sub_tag, sd, mode="train", exp_tag=resolved_exp_tag
        )
        raise FileNotFoundError(
            f"未找到数据集 {ds_name} (subset={sub_tag}, seed={sd}) 任何已训练阶段的检查点（如 {expected_dir / 'best.ckpt'}）。"
        )

    resolved_eval_sub = _resolve_eval_subdir(eval_subdir)
    stage_results: dict[str, dict] = {}
    for st, st_tag, st_ignore_imp in available_stages:
        stage_out_dir = (
            _apply_eval_subdir_to_seed_dir(
                workflow_output_directory(
                    ds_name,
                    st,
                    sub_tag,
                    sd,
                    mode="train",
                    exp_tag="",
                    ignore_implicit_tag=True,
                ),
                resolved_eval_sub,
                stage=st,
            )
            if st_ignore_imp
            else None
        )
        stage_results[st] = _evaluate_single_stage(
            ds_name,
            stage=st,
            seed=sd,
            split=split,
            subject=subject,
            num_subjects=num_subjects,
            subjects=subjects,
            eval_batch_size=eval_batch_size,
            num_workers=num_workers,
            precision=precision,
            window_seconds=window_seconds,
            window_start_offset_seconds=window_start_offset_seconds,
            require_cache=require_cache,
            output_dir=stage_out_dir,
            eval_subdir=resolved_eval_sub,
            exp_tag=st_tag,
            word_exp_tag=word_exp_tag,
            config_path=None,
            control_config_path=effective_control_config_path,
            shift_seconds=shift_seconds,
            shift_mode=shift_mode,
            preroll_seconds=preroll_seconds,
            cms_mode=cms_mode,
            cms_ema_decay=cms_ema_decay,
            zero_signal_control=zero_signal_control,
            subtraction_etas=subtraction_etas,
            plot=plot,
            show_progress=show_progress,
            max_eval_rows=max_eval_rows,
            verbose=verbose,
            print_report=False,
            common_horizon_delay=common_horizon_delay,
        )

    effective_summary_tag = (
        available_stages[0][1]
        if len(available_stages) == 1
        else resolved_exp_tag
    )
    dataset_eval_root = base_root / ds_name
    if effective_summary_tag:
        dataset_eval_root = dataset_eval_root / effective_summary_tag
    if resolved_eval_sub:
        dataset_eval_root = dataset_eval_root / resolved_eval_sub
    comp_dir = (
        Path(output_dir).resolve()
        if output_dir is not None
        else (
            dataset_eval_root
            / f"evaluation_{split}"
            / sub_tag
            / f"seed-{sd:03d}"
        ).resolve()
    )
    comp_dir.mkdir(parents=True, exist_ok=True)
    comp_figs = {}
    if plot:
        comp_figs = plot_stage_comparison_figure(
            stage_results, comp_dir / "figures", split=split
        )
        for st_res in stage_results.values():
            st_figs = plot_stage_comparison_figure(
                stage_results, Path(st_res["output_dir"]) / "figures", split=split
            )
            st_res["figures"].update(st_figs)
            save_json(Path(st_res["evaluation_json"]), st_res)

    stage_significance: dict[str, dict] = {}
    if "word" in stage_results:
        w_subs = (
            (stage_results["word"].get("metrics") or {}).get("per_subject")
            or stage_results["word"].get("per_subject")
            or []
        )
        w_map = {
            str(s.get("subject_id")): float(
                s.get("macro_top10", s.get("n50_macro_top10", 0.0))
            )
            for s in w_subs
            if s.get("subject_id") is not None
        }
        for st_comp, comp_key in (
            ("context", "context_vs_word_n50_top10"),
            ("revision", "revision_vs_word_n50_top10"),
            ("prefix_revision", "prefix_revision_vs_word_n50_top10"),
        ):
            if st_comp in stage_results:
                c_subs = (
                    (stage_results[st_comp].get("metrics") or {}).get(
                        "per_subject"
                    )
                    or stage_results[st_comp].get("per_subject")
                    or []
                )
                c_map = {
                    str(s.get("subject_id")): float(
                        s.get("macro_top10", s.get("n50_macro_top10", 0.0))
                    )
                    for s in c_subs
                    if s.get("subject_id") is not None
                }
                common_sids = sorted(set(w_map.keys()) & set(c_map.keys()))
                if common_sids:
                    a_arr = np.asarray(
                        [c_map[sid] for sid in common_sids], dtype=np.float64
                    )
                    b_arr = np.asarray(
                        [w_map[sid] for sid in common_sids], dtype=np.float64
                    )
                    sig_res = _compute_paired_block_significance(
                        a_arr,
                        b_arr,
                        [str(sid) for sid in common_sids],
                        label=f"{st_comp} vs word (N50 Macro Top-10)",
                        metric="vocab50_macro_top10",
                        seed=20261029,
                    )
                    stage_significance[comp_key] = sig_res
                    stage_significance[
                        f"{st_comp}_minus_word_vocab50_top10"
                    ] = sig_res

    multi_summary = {
        "status": "multi_stage_evaluation_completed",
        "dataset": ds_name,
        "exp_tag": effective_summary_tag,
        "word_exp_tag": resolved_word_exp_tag
        if has_explicit_word_tag
        else resolved_exp_tag,
        "split": split,
        "seed": sd,
        "window_seconds": float(base_cfg["workflow"]["window_seconds"]),
        "window_samples": int(base_cfg["workflow"]["window_samples"]),
        "window_start_offset_seconds": float(
            base_cfg["workflow"]["window_start_offset_seconds"]
        ),
        "subset": subset_spec,
        "evaluated_stages": [st for st, _, _ in available_stages],
        "stages": stage_results,
        "stage_significance": stage_significance,
        "output_dir": str(comp_dir),
        "figures": comp_figs,
    }
    comp_exports = export_stage_comparison_tables_and_markdown(
        multi_summary, comp_dir
    )
    multi_summary["exports"] = comp_exports
    summary_file = comp_dir / f"multi_stage_evaluation_{split}.json"
    save_json(summary_file, multi_summary)
    multi_summary["evaluation_json"] = str(summary_file)
    print_stage_comparison_report(
        multi_summary, split=split, verbose=verbose, output_dir=comp_dir
    )
    if _export_outer_multi_seed:
        for st_res in stage_results.values():
            st_outer_dir = Path(str(st_res["output_dir"])).resolve().parent
            summarize_multi_seed_evaluation(
                [st_res],
                split=split,
                output_dir=st_outer_dir,
                verbose=verbose,
                print_report=False,
                control_method=control_method,
                control_metric=control_metric,
            )
        outer_comp_dir = comp_dir.parent
        outer_comp_summary = summarize_multi_seed_evaluation(
            [multi_summary],
            split=split,
            output_dir=outer_comp_dir,
            verbose=verbose,
            print_report=True,
            control_method=control_method,
            control_metric=control_metric,
        )
        multi_summary["outer_multi_seed_evaluation_json"] = str(
            outer_comp_summary.get("evaluation_json", "")
        )
        multi_summary["outer_multi_seed_exports"] = outer_comp_summary.get(
            "exports", {}
        )
    return multi_summary
