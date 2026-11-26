"""单阶段词级检查点留出集评估执行器（Clean + 时间错位 / 零信号负对照 + Common-Horizon 子集）。"""

from __future__ import annotations

import gc
import json
import os
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import torch

from braindecoding.config import project_path
from braindecoding.evaluation.word_retrieval_common_horizon import (
    _apply_eval_subdir_to_seed_dir,
    _evaluate_common_horizon_subset_from_pt,
    _is_encoded_cache_compatible,
    _resolve_eval_subdir,
)
from braindecoding.evaluation.word_retrieval_controls import (
    _extract_control_summary_row,
    _format_shift_label,
    _resolve_control_specification,
    _summarize_shifted_conditions,
)
from braindecoding.evaluation.word_retrieval_metrics import (
    _compute_encoded_evaluation_metrics,
)
from braindecoding.evaluation.word_retrieval_statistics import (
    STAGE_DISPLAY_NAMES,
    export_word_evaluation_tables_and_markdown,
    plot_evaluation_figures,
    print_word_evaluation_report,
)
from braindecoding.experiment import file_sha256
from braindecoding.models import build_brain_embedding_model
from braindecoding.training import meg_word_task
from braindecoding.training.protocol_stage_training import group_column
from braindecoding.training.training_runtime import (
    choose_device,
    limit_rows,
    load_checkpoint,
    save_json,
    set_seed,
)
from braindecoding.training.word_encoder_training import (
    encode_loader,
    make_loader,
    resolve_amp_dtype,
)

def _evaluate_single_stage(
    dataset: str | None = None,
    *,
    stage: str | None = None,
    seed: int | None = 0,
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
    plot: bool = True,
    show_progress: bool = True,
    max_eval_rows: int | None = None,
    verbose: bool = False,
    print_report: bool = True,
    common_horizon_delay: int | None = None,
    synthetic_timing: str | None = None,
) -> dict:
    """对单个训练阶段（word / context / revision）执行验证集或测试集评估并生成图表。"""
    from braindecoding.training.workflow_runner import (
        build_workflow_config,
        run_cache_workflow,
        verify_cache_ready,
        workflow_output_directory,
    )

    if split not in {"val", "test"}:
        raise ValueError(f"评估 split 只能是 'val' 或 'test'，收到：{split!r}")

    cfg, subset_spec = build_workflow_config(
        dataset,
        stage=stage,
        seed=seed,
        subject=subject,
        num_subjects=num_subjects,
        subjects=subjects,
        eval_batch_size=eval_batch_size,
        num_workers=num_workers,
        precision=precision,
        window_seconds=window_seconds,
        window_start_offset_seconds=window_start_offset_seconds,
        shift_mode=shift_mode,
        preroll_seconds=preroll_seconds,
        mode="train",
        output_dir=output_dir,
        exp_tag=exp_tag,
        word_exp_tag=word_exp_tag,
        config_path=config_path,
        synthetic_timing=synthetic_timing,
    )
    control_spec = _resolve_control_specification(
        cfg,
        control_config_path=control_config_path,
        shift_seconds=shift_seconds,
        zero_signal_control=zero_signal_control,
        subtraction_etas=subtraction_etas,
        shift_mode=shift_mode,
        preroll_seconds=preroll_seconds,
        cms_mode=cms_mode,
        cms_ema_decay=cms_ema_decay,
    )
    dataset = subset_spec["dataset"]
    stage = cfg["workflow"]["stage"]
    resolved_exp_tag = cfg["workflow"].get("exp_tag", "")
    resolved_word_exp_tag = cfg["workflow"].get(
        "word_exp_tag", resolved_exp_tag
    )
    tc = cfg["training"]
    seed = int(tc["seed"])
    out = Path(tc["output_dir"])

    if checkpoint_path is not None:
        ckpt_file = Path(checkpoint_path).resolve()
        if not ckpt_file.is_file():
            raise FileNotFoundError(f"指定的评估检查点不存在：{ckpt_file}")
        if output_dir is None:
            out = ckpt_file.parent
    else:
        searched_ckpt_paths: list[Path] = [out / "best.ckpt"]
        ckpt_file = out / "best.ckpt"
        if not ckpt_file.is_file():
            ckpt_file = out / "best.pt"
        if not ckpt_file.is_file():
            smoke_dir = workflow_output_directory(
                dataset,
                stage,
                subset_spec["subset_tag"],
                seed,
                mode="smoke",
                exp_tag=resolved_exp_tag,
            )
            smoke_ckpt = smoke_dir / "best.ckpt"
            if not smoke_ckpt.is_file():
                smoke_ckpt = smoke_dir / "best.pt"
            if smoke_ckpt.is_file():
                ckpt_file = smoke_ckpt
                if output_dir is None:
                    out = smoke_dir
        if not ckpt_file.is_file() and stage == "word":
            fallback_tags: list[tuple[str, bool]] = []
            if resolved_word_exp_tag != resolved_exp_tag:
                fallback_tags.append(
                    (resolved_word_exp_tag, resolved_word_exp_tag == "")
                )
            if resolved_exp_tag and ("", True) not in fallback_tags:
                fallback_tags.append(("", True))
            for cand_tag, cand_ignore_imp in fallback_tags:
                for cand_mode in ("train", "smoke"):
                    cand_dir = workflow_output_directory(
                        dataset,
                        stage,
                        subset_spec["subset_tag"],
                        seed,
                        mode=cand_mode,
                        exp_tag=cand_tag,
                        ignore_implicit_tag=cand_ignore_imp,
                    )
                    if cand_mode == "train" and (cand_dir / "best.ckpt") not in searched_ckpt_paths:
                        searched_ckpt_paths.append(cand_dir / "best.ckpt")
                    cand_ckpt = cand_dir / "best.ckpt"
                    if not cand_ckpt.is_file():
                        cand_ckpt = cand_dir / "best.pt"
                    if cand_ckpt.is_file():
                        ckpt_file = cand_ckpt
                        resolved_exp_tag = cand_tag
                        if output_dir is None:
                            out = cand_dir
                        break
                if ckpt_file.is_file():
                    break
        if not ckpt_file.is_file():
            searched_str = " 或 ".join(str(p) for p in searched_ckpt_paths)
            raise FileNotFoundError(
                f"未找到阶段 {stage} ({STAGE_DISPLAY_NAMES.get(stage, stage)}) 的训练检查点：{searched_str}。"
                f"请先完成该阶段的训练，或检查 EXP_TAG / WORD_EXP_TAG，或通过 --checkpoint 显式指定检查点路径。"
            )

    if output_dir is None:
        out = _apply_eval_subdir_to_seed_dir(out, eval_subdir, stage=stage)
    out.mkdir(parents=True, exist_ok=True)

    encoded_cache_pt = (
        out / "exploratory" / f"subspace_encoded_cache_{split}.pt"
    )
    if (
        common_horizon_delay is not None
        and int(common_horizon_delay) > 0
        and stage in ("word", "context")
    ):
        has_isolated_eval_subdir = bool(_resolve_eval_subdir(eval_subdir))
        resolved_ckpt_file = ckpt_file if ckpt_file.is_file() else None
        if not _is_encoded_cache_compatible(
            encoded_cache_pt,
            control_spec=control_spec,
            ckpt_file=resolved_ckpt_file,
        ):
            _evaluate_single_stage(
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
                plot=(plot and not has_isolated_eval_subdir),
                show_progress=show_progress,
                max_eval_rows=max_eval_rows,
                verbose=verbose,
                print_report=False,
                common_horizon_delay=None,
            )
        if encoded_cache_pt.is_file():
            return _evaluate_common_horizon_subset_from_pt(
                encoded_cache_pt,
                cfg=cfg,
                subset_spec=subset_spec,
                control_spec=control_spec,
                common_horizon_delay=int(common_horizon_delay),
                ckpt_file=resolved_ckpt_file,
                output_dir=out,
                split=split,
                verbose=verbose,
                print_report=print_report,
                promote_to_primary=has_isolated_eval_subdir,
                plot=(plot and has_isolated_eval_subdir),
            )

    if require_cache:
        cache_verification = verify_cache_ready(cfg, splits=(split,))
    else:
        try:
            cache_verification = verify_cache_ready(cfg, splits=(split,))
        except FileNotFoundError:
            run_cache_workflow(
                dataset,
                subject=subject,
                num_subjects=num_subjects,
                subjects=subjects,
                include_test=(split == "test"),
                force=False,
                config_path=config_path,
            )
            cache_verification = verify_cache_ready(cfg, splits=(split,))

    device = choose_device(tc.get("device", "auto"))
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    set_seed(seed)
    checkpoint = load_checkpoint(ckpt_file, map_location="cpu")
    ckpt_cfg_ds_all = (
        checkpoint.get("config", {}).get("dataset", {})
        if isinstance(checkpoint.get("config"), dict)
        else {}
    )
    if (
        synthetic_timing is None
        and not cfg["dataset"].get("synthetic_timing")
        and isinstance(ckpt_cfg_ds_all, dict)
        and ckpt_cfg_ds_all.get("synthetic_timing")
    ):
        cfg["dataset"]["synthetic_timing"] = str(ckpt_cfg_ds_all["synthetic_timing"]).strip().lower()
        cfg["workflow"]["synthetic_timing"] = cfg["dataset"]["synthetic_timing"]
    if not bool(cfg["workflow"].get("window_explicitly_overridden", False)):
        ckpt_cfg_ds = (
            checkpoint.get("config", {}).get("dataset", {})
            if isinstance(checkpoint.get("config"), dict)
            else {}
        )
        ckpt_win = checkpoint.get(
            "window_seconds",
            ckpt_cfg_ds.get("window_seconds")
            if isinstance(ckpt_cfg_ds, dict)
            else None,
        )
        ckpt_off = checkpoint.get(
            "window_start_offset_seconds",
            ckpt_cfg_ds.get("window_start_offset_seconds")
            if isinstance(ckpt_cfg_ds, dict)
            else None,
        )
        ckpt_base = checkpoint.get(
            "baseline_seconds",
            ckpt_cfg_ds.get("baseline_seconds")
            if isinstance(ckpt_cfg_ds, dict)
            else None,
        )
        ckpt_hp = checkpoint.get(
            "causal_highpass_seconds",
            ckpt_cfg_ds.get("causal_highpass_seconds")
            if isinstance(ckpt_cfg_ds, dict)
            else None,
        )
        ckpt_hp_mode = checkpoint.get(
            "causal_highpass_mode",
            ckpt_cfg_ds.get("causal_highpass_mode")
            if isinstance(ckpt_cfg_ds, dict)
            else None,
        )
        ckpt_hp_hz = checkpoint.get(
            "causal_highpass_hz",
            ckpt_cfg_ds.get("causal_highpass_hz")
            if isinstance(ckpt_cfg_ds, dict)
            else None,
        )
        ckpt_hp_order = checkpoint.get(
            "causal_highpass_order",
            ckpt_cfg_ds.get("causal_highpass_order")
            if isinstance(ckpt_cfg_ds, dict)
            else None,
        )
        ckpt_clamp = checkpoint.get(
            "clamp",
            ckpt_cfg_ds.get("clamp")
            if isinstance(ckpt_cfg_ds, dict)
            else None,
        )
        if ckpt_win is not None:
            cfg["dataset"]["window_seconds"] = float(ckpt_win)
            cfg["workflow"]["window_seconds"] = float(ckpt_win)
            sr_hz = float(cfg["workflow"].get("sampling_rate_hz", 50.0))
            cfg["workflow"]["window_samples"] = int(round(float(ckpt_win) * sr_hz))
        if ckpt_off is not None:
            cfg["dataset"]["window_start_offset_seconds"] = float(ckpt_off)
            cfg["workflow"]["window_start_offset_seconds"] = float(ckpt_off)
        if ckpt_base is not None:
            cfg["dataset"]["baseline_seconds"] = float(ckpt_base)
        if ckpt_hp is not None:
            cfg["dataset"]["causal_highpass_seconds"] = float(ckpt_hp)
        if ckpt_hp_mode is not None:
            cfg["dataset"]["causal_highpass_mode"] = str(ckpt_hp_mode)
        if ckpt_hp_hz is not None:
            cfg["dataset"]["causal_highpass_hz"] = float(ckpt_hp_hz)
        if ckpt_hp_order is not None:
            cfg["dataset"]["causal_highpass_order"] = int(ckpt_hp_order)
        if ckpt_clamp is not None:
            cfg["dataset"]["clamp"] = float(ckpt_clamp)

    read_fn = meg_word_task.load_event_table
    build_fn = meg_word_task.build_dataset
    selected_subjects = cfg["dataset"]["subjects"]

    train_table = read_fn(
        cfg["cache"]["event_table"],
        split="train",
        subjects=selected_subjects,
        trainable_only=True,
    )
    eval_table = read_fn(
        cfg["cache"]["event_table"],
        split=split,
        subjects=selected_subjects,
        trainable_only=True,
    )
    if max_eval_rows is not None:
        eval_table = limit_rows(eval_table, int(max_eval_rows))
    if len(eval_table) <= 0:
        raise ValueError(
            f"数据集 {dataset} 在划分 {split!r} 与受试者 {selected_subjects} 上的事件数为 0。"
        )

    eval_ds = build_fn(cfg, eval_table)
    grp_col = group_column(cfg)
    eval_bs = int(tc.get("eval_batch_size", tc["batch_size"]))
    eval_loader, _ = make_loader(
        eval_ds,
        tc,
        shuffle=False,
        group_column=grp_col,
        batch_size=eval_bs,
    )

    train_word_col = (
        "normalized_word"
        if "normalized_word" in train_table.columns
        else "标准词"
    )
    vocabulary = sorted(set(train_table[train_word_col].astype(str)))
    vocab50 = list(
        meg_word_task.frozen_vocabulary(
            dataset=dataset,
            train_words=train_table[train_word_col].astype(str).tolist(),
        )
    )

    model_cfg = checkpoint.get("model_config", cfg["model"])
    ckpt_subjects = list(checkpoint.get("subject_ids", selected_subjects))
    subject_count = max(len(ckpt_subjects), eval_ds.subject_count)

    model = build_brain_embedding_model(
        eval_ds.channel_count,
        eval_ds.channel_positions,
        subject_count,
        model_cfg,
    ).to(device)
    model.load_state_dict(checkpoint["model_state"], strict=True)

    amp_enabled = bool(tc.get("amp", True)) and device.type == "cuda"
    amp_dtype = resolve_amp_dtype(tc)
    encoded = encode_loader(
        model,
        eval_loader,
        device,
        amp=amp_enabled,
        amp_dtype=amp_dtype,
        show_progress=show_progress,
        progress_desc=f"Eval [{stage} | {split} | Clean 0s]",
    )

    metrics = _compute_encoded_evaluation_metrics(
        encoded,
        eval_ds,
        vocabulary,
        vocab50,
        cfg,
        stage=stage,
        split=split,
    )

    negative_controls_result: dict = {
        "specification": control_spec,
        "clean_reference": _extract_control_summary_row(
            "clean_0s", "Clean (0s)", 0.0, metrics
        ),
        "conditions": [],
        "shifted_aggregate": {},
    }
    cms_cfg = control_spec.get("common_mode_subtraction") or {}
    cms_enabled = bool(cms_cfg.get("enabled", False))
    cms_mode_str = str(cms_cfg.get("mode", "leave_one_out")).strip().lower()
    ctrl_shift_mode = str(
        control_spec.get("shift_mode", "deterministic_circular")
    )
    ctrl_preroll_seconds = float(control_spec.get("preroll_seconds", 45.0))
    sub_cfg = control_spec.get("subspace_coherence_debiasing") or {}
    sub_enabled = bool(sub_cfg.get("enabled", False))
    keep_shifted_encoded = (
        cms_enabled
        or sub_enabled
        or len(control_spec.get("shift_seconds", ())) >= 2
    )
    shifted_encoded_by_shift: list[tuple[float, str, str, dict]] = []
    shifted_ds_by_shift: dict[float, object] = {}

    if control_spec["enabled"]:
        clean_summary = negative_controls_result["clean_reference"]
        shifted_summaries: list[dict] = []
        for shift_sec in control_spec["shift_seconds"]:
            cond_key, cond_label = _format_shift_label(float(shift_sec))
            shift_ds = build_fn(
                cfg,
                eval_table,
                temporal_shift_seconds=float(shift_sec),
                shift_mode=ctrl_shift_mode,
                preroll_seconds=ctrl_preroll_seconds,
                preroll_source_table=(
                    train_table if ctrl_shift_mode == "preroll" else None
                ),
            )
            shift_loader, _ = make_loader(
                shift_ds,
                tc,
                shuffle=False,
                group_column=grp_col,
                batch_size=eval_bs,
            )
            shift_encoded = encode_loader(
                model,
                shift_loader,
                device,
                amp=amp_enabled,
                amp_dtype=amp_dtype,
                show_progress=show_progress,
                progress_desc=f"Eval [{stage} | {split} | {cond_label}]",
            )
            shift_metrics = _compute_encoded_evaluation_metrics(
                shift_encoded,
                shift_ds,
                vocabulary,
                vocab50,
                cfg,
                stage=stage,
                split=split,
            )
            summary_row = _extract_control_summary_row(
                cond_key,
                cond_label,
                float(shift_sec),
                shift_metrics,
                clean_summary=clean_summary,
            )
            shifted_summaries.append(summary_row)
            negative_controls_result["conditions"].append(
                {
                    "key": cond_key,
                    "label": cond_label,
                    "control_type": "temporal_shift",
                    "shift_seconds": float(shift_sec),
                    "summary": summary_row,
                    "metrics": {
                        "vocabulary_sweep": shift_metrics.get("vocabulary_sweep", {}),
                        "revision_horizons": shift_metrics.get("revision_horizons", {}),
                    },
                }
            )
            if keep_shifted_encoded:
                shifted_encoded_by_shift.append(
                    (float(shift_sec), cond_key, cond_label, shift_encoded)
                )
                if cms_enabled:
                    shifted_ds_by_shift[float(shift_sec)] = shift_ds
                else:
                    del shift_ds
                del shift_loader
            else:
                del shift_ds, shift_loader, shift_encoded

        if control_spec.get("include_zero_signal", False):
            zero_ds = build_fn(cfg, eval_table, zero_signal=True)
            zero_loader, _ = make_loader(
                zero_ds,
                tc,
                shuffle=False,
                group_column=grp_col,
                batch_size=eval_bs,
            )
            zero_encoded = encode_loader(
                model,
                zero_loader,
                device,
                amp=amp_enabled,
                amp_dtype=amp_dtype,
                show_progress=show_progress,
                progress_desc=f"Eval [{stage} | {split} | Zero Signal]",
            )
            zero_metrics = _compute_encoded_evaluation_metrics(
                zero_encoded,
                zero_ds,
                vocabulary,
                vocab50,
                cfg,
                stage=stage,
                split=split,
            )
            zero_summary = _extract_control_summary_row(
                "zero_signal",
                "Zero Signal",
                None,
                zero_metrics,
                clean_summary=clean_summary,
            )
            negative_controls_result["conditions"].append(
                {
                    "key": "zero_signal",
                    "label": "Zero Signal",
                    "control_type": "zero_signal",
                    "shift_seconds": None,
                    "summary": zero_summary,
                    "metrics": {
                        "vocabulary_sweep": zero_metrics.get("vocabulary_sweep", {}),
                        "revision_horizons": zero_metrics.get("revision_horizons", {}),
                    },
                }
            )
            del zero_ds, zero_loader, zero_encoded

        if shifted_summaries:
            negative_controls_result["shifted_aggregate"] = (
                _summarize_shifted_conditions(clean_summary, shifted_summaries)
            )
        elif negative_controls_result["conditions"]:
            cond_summaries = [
                c["summary"]
                for c in negative_controls_result["conditions"]
                if isinstance(c.get("summary"), dict)
            ]
            if cond_summaries:
                negative_controls_result["shifted_aggregate"] = (
                    _summarize_shifted_conditions(clean_summary, cond_summaries)
                )

    mu0_payload = checkpoint.get("causal_cms_mu0")
    if (
        cms_enabled
        and cms_mode_str in {"causal_ema", "pos_causal_ema"}
        and not isinstance(mu0_payload, (Mapping, torch.Tensor))
    ):
        from braindecoding.training.siamese_debiased_training import (
            compute_causal_cms_mu0_from_encoded,
        )

        mu0_train_table = (
            limit_rows(train_table, int(max_eval_rows))
            if max_eval_rows is not None
            else train_table
        )
        train_mu0_ds = build_fn(cfg, mu0_train_table)
        train_mu0_loader, _ = make_loader(
            train_mu0_ds,
            tc,
            shuffle=False,
            group_column=grp_col,
            batch_size=eval_bs,
        )
        train_mu0_enc = encode_loader(
            model,
            train_mu0_loader,
            device,
            amp=amp_enabled,
            amp_dtype=amp_dtype,
            show_progress=show_progress,
            progress_desc=f"Eval [{stage} | train mu_0 calibration]",
        )
        mu0_payload = compute_causal_cms_mu0_from_encoded(
            train_mu0_enc, source="train_readonly_fallback"
        )
        del train_mu0_ds, train_mu0_loader, train_mu0_enc

    ckpt_epoch = int(checkpoint.get("epoch", 0))
    ckpt_selection = str(checkpoint.get("selection", "legacy50"))
    del model, eval_loader, checkpoint
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    training_history = []
    summary_path = out / "training_summary.json"
    if not summary_path.is_file() and ckpt_file.parent.is_dir():
        summary_path = ckpt_file.parent / "training_summary.json"
    if summary_path.is_file():
        try:
            summary_data = json.loads(summary_path.read_text(encoding="utf-8"))
            training_history = list(summary_data.get("history", ()))
        except Exception:
            training_history = []

    out.mkdir(parents=True, exist_ok=True)
    eval_win_sec = float(
        getattr(
            eval_ds, "window_seconds", cfg["dataset"].get("window_seconds", 1.0)
        )
    )
    eval_win_samp = int(
        getattr(eval_ds, "window_samples", round(eval_win_sec * 50.0))
    )
    eval_win_off = float(
        cfg["dataset"].get("window_start_offset_seconds", 0.0)
    )
    eval_result = {
        "status": "evaluation_completed",
        "dataset": dataset,
        "stage": stage,
        "exp_tag": resolved_exp_tag,
        "word_exp_tag": resolved_word_exp_tag,
        "stage_display_name": STAGE_DISPLAY_NAMES.get(stage, stage),
        "split": split,
        "seed": seed,
        "synthetic_timing": cfg["dataset"].get("synthetic_timing"),
        "precision": tc["precision"],
        "window_seconds": eval_win_sec,
        "window_samples": eval_win_samp,
        "window_start_offset_seconds": eval_win_off,
        "subset": subset_spec,
        "checkpoint": {
            "path": str(ckpt_file),
            "sha256": file_sha256(ckpt_file),
            "epoch": ckpt_epoch,
            "selection": ckpt_selection,
        },
        "cache_verification": cache_verification,
        "metrics": metrics,
        "negative_controls": negative_controls_result,
        "training_history": training_history,
        "output_dir": str(out),
        "figures": {},
    }

    print_word_evaluation_report(
        eval_result,
        verbose=verbose,
        output_dir=out,
        print_to_stdout=print_report,
    )

    if plot:
        figs = plot_evaluation_figures(eval_result, out / "figures")
        eval_result["figures"].update(figs)

    shifted_encoded_by_shift.clear()
    shifted_ds_by_shift.clear()

    eval_result["exported_files"] = export_word_evaluation_tables_and_markdown(
        eval_result, out
    )
    eval_json_path = out / f"evaluation_{split}.json"
    save_json(eval_json_path, eval_result)
    eval_result["evaluation_json"] = str(eval_json_path)

    del eval_ds, encoded
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return eval_result


evaluate_single_stage = _evaluate_single_stage
