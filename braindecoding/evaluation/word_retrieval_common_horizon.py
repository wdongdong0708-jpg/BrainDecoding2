"""等事件共同视界（common_horizon_delay）子集过滤、.pt 缓存兼容校验与隔离子目录复评。"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch

from braindecoding.config import project_path
from braindecoding.data.word_window_dataset import add_runtime_context_uid
from braindecoding.evaluation.word_retrieval_controls import (
    _extract_control_summary_row,
    _summarize_shifted_conditions,
)
from braindecoding.evaluation.word_retrieval_metrics import (
    _compute_encoded_evaluation_metrics,
)
from braindecoding.evaluation.word_retrieval_statistics import (
    STAGE_DISPLAY_NAMES,
    _CachedEvalDatasetProxy,
    export_word_evaluation_tables_and_markdown,
    plot_evaluation_figures,
    print_word_evaluation_report,
)
from braindecoding.experiment import file_sha256
from braindecoding.training import meg_word_task
from braindecoding.training.training_runtime import load_checkpoint, save_json

def _slice_encoded_dict(encoded: dict, keep_mask_np: np.ndarray) -> dict:
    """按布尔掩码 keep_mask_np 过滤已编码特征字典中的所有事件级张量与列表。"""
    keep_mask_np = np.asarray(keep_mask_np, dtype=bool)
    keep_mask_t = torch.from_numpy(keep_mask_np)
    keep_list = keep_mask_np.tolist()
    n_total = len(encoded["words"])
    sliced: dict[str, object] = {
        "predictions": encoded["predictions"][keep_mask_t],
        "targets": encoded["targets"][keep_mask_t],
        "words": [
            str(w) for w, keep in zip(encoded["words"], keep_list) if keep
        ],
        "event_ids": [
            str(eid)
            for eid, keep in zip(encoded["event_ids"], keep_list)
            if keep
        ],
    }
    rec_ids = encoded.get("recording_ids")
    if isinstance(rec_ids, Sequence) and len(rec_ids) == n_total:
        sliced["recording_ids"] = [
            str(rid) for rid, keep in zip(rec_ids, keep_list) if keep
        ]
    else:
        sliced["recording_ids"] = list(rec_ids or ())
    for mask_key in (
        "subject_indices",
        "sentence_indices",
        "local_context_internal_mask",
        "three_state_internal_mask",
    ):
        t_val = encoded.get(mask_key)
        if isinstance(t_val, torch.Tensor) and len(t_val) == n_total:
            sliced[mask_key] = t_val[keep_mask_t]
    if isinstance(encoded.get("revision_predictions"), dict):
        sliced["revision_predictions"] = {
            int(d): t[keep_mask_t]
            for d, t in encoded["revision_predictions"].items()
            if isinstance(t, torch.Tensor) and len(t) == n_total
        }
    if isinstance(encoded.get("revision_availability_masks"), dict):
        sliced["revision_availability_masks"] = {
            int(d): m[keep_mask_t]
            for d, m in encoded["revision_availability_masks"].items()
            if isinstance(m, torch.Tensor) and len(m) == n_total
        }
    return sliced


def _build_common_horizon_event_filter(
    cfg: dict,
    *,
    dataset: str,
    split: str,
    selected_subjects: Sequence[str],
    common_horizon_delay: int,
    fallback_eval_table=None,
) -> tuple[set[str], dict[str, str]]:
    """计算在同一语境分组内至少拥有 common_horizon_delay 个未来词的共同有效事件 ID 集合（如 Pallier2025 D=2 时 N=14,590, 50词查询=7,440）。"""
    delay_int = max(0, int(common_horizon_delay))
    kept_event_ids: set[str] = set()
    event_id_to_block: dict[str, str] = {}

    read_fn = meg_word_task.load_event_table
    source_df = None
    try:
        ev_path = cfg.get("cache", {}).get("event_table")
        if ev_path is not None:
            loaded_df = read_fn(
                ev_path,
                split=split,
                subjects=selected_subjects,
                trainable_only=True,
            )
            if len(loaded_df) > 0:
                if "runtime_context_uid" not in loaded_df.columns:
                    loaded_df = add_runtime_context_uid(loaded_df)
                source_df = loaded_df
    except Exception:
        source_df = None

    if source_df is None and fallback_eval_table is not None:
        if any(
            col in fallback_eval_table.columns
            for col in (
                "runtime_context_uid",
                "sentence_uid",
                "context_id",
                "上下文编号",
            )
        ):
            source_df = fallback_eval_table.copy()
            if (
                dataset.startswith("pallier2025")
                and "runtime_context_uid" not in source_df.columns
                and "recording_id" in source_df.columns
                and "sentence_uid" in source_df.columns
            ):
                source_df = add_runtime_context_uid(source_df)

    eid_col = (
        None
        if source_df is None
        else (
            "event_id"
            if "event_id" in source_df.columns
            else ("事件编号" if "事件编号" in source_df.columns else None)
        )
    )
    if source_df is not None and eid_col is not None:
        grp_col = next(
            (
                col
                for col in (
                    "runtime_context_uid",
                    "sentence_uid",
                    "context_id",
                    "上下文编号",
                    "recording_id",
                    "记录编号",
                )
                if col in source_df.columns
            ),
            None,
        )
        blk_col = next(
            (
                col
                for col in (
                    "sentence_uid",
                    "上下文编号",
                    "context_id",
                    "runtime_context_uid",
                    "recording_id",
                )
                if col in source_df.columns
            ),
            grp_col,
        )
        if grp_col is not None:
            for _, grp_df in source_df.groupby(grp_col, sort=False):
                eids = grp_df[eid_col].astype(str).tolist()
                blks = (
                    grp_df[blk_col].astype(str).tolist()
                    if blk_col is not None and blk_col in grp_df.columns
                    else [str(grp_df[grp_col].iloc[0])] * len(eids)
                )
                n_grp = len(eids)
                for pos, (eid, blk_val) in enumerate(zip(eids, blks)):
                    event_id_to_block[eid] = blk_val
                    if pos + delay_int < n_grp:
                        kept_event_ids.add(eid)
            if kept_event_ids:
                return kept_event_ids, event_id_to_block

    # Fallback: 从同数据集已有的 prefix_revision / revision 特征缓存 .pt 中直接提取 availability mask
    from braindecoding.config import results_root

    ds_root = results_root() / dataset
    if ds_root.is_dir():
        for rev_st in ("prefix_revision", "revision"):
            for pt_cand in sorted(
                ds_root.rglob(
                    f"{rev_st}/**/exploratory/subspace_encoded_cache_{split}.pt"
                ),
                key=lambda p: p.stat().st_mtime if p.exists() else 0.0,
                reverse=True,
            ):
                try:
                    b_cand = torch.load(
                        pt_cand, map_location="cpu", weights_only=False
                    )
                    c_enc = b_cand.get("clean_encoded") or {}
                    masks = c_enc.get("revision_availability_masks") or {}
                    mask_t = masks.get(delay_int)
                    if mask_t is None and str(delay_int) in masks:
                        mask_t = masks[str(delay_int)]
                    eids = c_enc.get("event_ids") or []
                    if (
                        isinstance(mask_t, torch.Tensor)
                        and len(eids) == len(mask_t)
                    ):
                        for eid, keep in zip(
                            eids, mask_t.detach().cpu().tolist()
                        ):
                            if bool(keep):
                                kept_event_ids.add(str(eid))
                        if kept_event_ids:
                            return kept_event_ids, event_id_to_block
                except Exception:
                    continue

    raise RuntimeError(
        f"无法构造数据集 {dataset} (split={split}, delay={delay_int}) 的共同有效事件掩码："
        f"请确保已解包派生事件表缓存或同目录下存在 prefix_revision 特征缓存。"
    )


def _evaluate_common_horizon_subset_from_pt(
    encoded_cache_pt: Path,
    *,
    cfg: dict,
    subset_spec: dict,
    control_spec: dict,
    common_horizon_delay: int,
    ckpt_file: Path | None,
    output_dir: Path,
    split: str,
    verbose: bool = False,
    print_report: bool = True,
    promote_to_primary: bool = False,
    plot: bool = False,
) -> dict:
    """直接从已编码缓存过滤并评估共同有效视界子集，不覆盖全量评估文件。"""
    cached_bundle = torch.load(
        encoded_cache_pt, map_location="cpu", weights_only=False
    )
    dataset = str(subset_spec["dataset"])
    stage = str(cfg["workflow"]["stage"])
    resolved_exp_tag = str(
        cached_bundle.get("exp_tag", cfg["workflow"].get("exp_tag", ""))
    )
    resolved_word_exp_tag = str(
        cached_bundle.get(
            "word_exp_tag",
            cfg["workflow"].get("word_exp_tag", resolved_exp_tag),
        )
    )
    tc = cfg["training"]
    seed = int(cached_bundle.get("seed", tc["seed"]))
    selected_subjects = cfg["dataset"]["subjects"]

    kept_event_ids, event_id_to_block = _build_common_horizon_event_filter(
        cfg,
        dataset=dataset,
        split=split,
        selected_subjects=selected_subjects,
        common_horizon_delay=int(common_horizon_delay),
        fallback_eval_table=cached_bundle.get("eval_table"),
    )

    raw_clean = cached_bundle["clean_encoded"]
    keep_mask_np = np.asarray(
        [str(eid) in kept_event_ids for eid in raw_clean["event_ids"]],
        dtype=bool,
    )
    if not keep_mask_np.any():
        raise ValueError(
            f"在 {encoded_cache_pt} 中应用 common_horizon_delay={common_horizon_delay} 后保留事件数为 0。"
        )

    clean_encoded = _slice_encoded_dict(raw_clean, keep_mask_np)
    shifted_encoded_by_shift = [
        (
            float(sec),
            str(k),
            str(lbl),
            _slice_encoded_dict(enc, keep_mask_np),
        )
        for sec, k, lbl, enc in (
            cached_bundle.get("shifted_encoded_by_shift") or ()
        )
    ]

    raw_table = cached_bundle["eval_table"].copy()
    raw_table["_eid_str"] = raw_table["event_id"].astype(str)
    dedup_table = raw_table.drop_duplicates(subset=["_eid_str"]).set_index(
        "_eid_str"
    )
    eval_table_sub = dedup_table.reindex(clean_encoded["event_ids"]).reset_index(
        drop=True
    )
    eval_table_sub["event_id"] = list(clean_encoded["event_ids"])
    if (
        event_id_to_block
        and "sentence_uid" not in eval_table_sub.columns
        and "上下文编号" not in eval_table_sub.columns
    ):
        eval_table_sub["sentence_uid"] = [
            event_id_to_block.get(str(eid), f"blk-{idx // 8:04d}")
            for idx, eid in enumerate(clean_encoded["event_ids"])
        ]

    eval_win_sec = float(
        cached_bundle.get(
            "window_seconds", cfg["dataset"].get("window_seconds", 1.0)
        )
    )
    eval_win_samp = int(
        cached_bundle.get("window_samples", round(eval_win_sec * 50.0))
    )
    eval_win_off = float(
        cfg["dataset"].get("window_start_offset_seconds", 0.0)
    )
    proxy_ds = _CachedEvalDatasetProxy(
        eval_table_sub,
        cached_bundle["embedding_map"],
        window_seconds=eval_win_sec,
        window_samples=eval_win_samp,
    )
    vocabulary = [str(w) for w in cached_bundle["vocabulary"]]
    vocab50 = [str(w) for w in cached_bundle["vocab50"]]

    metrics = _compute_encoded_evaluation_metrics(
        clean_encoded,
        proxy_ds,
        vocabulary,
        vocab50,
        cfg,
        stage=stage,
        split=split,
    )

    clean_summary = _extract_control_summary_row(
        "clean_0s", "Clean (0s)", 0.0, metrics
    )
    negative_controls_result: dict[str, object] = {
        "specification": control_spec,
        "clean_reference": clean_summary,
        "conditions": [],
        "shifted_aggregate": {},
    }
    shifted_summaries: list[dict] = []
    for shift_sec, cond_key, cond_label, shift_enc in shifted_encoded_by_shift:
        shift_metrics = _compute_encoded_evaluation_metrics(
            shift_enc,
            proxy_ds,
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
                    "vocabulary_sweep": shift_metrics.get(
                        "vocabulary_sweep", {}
                    ),
                    "revision_horizons": shift_metrics.get(
                        "revision_horizons", {}
                    ),
                },
            }
        )
    if shifted_summaries:
        negative_controls_result["shifted_aggregate"] = (
            _summarize_shifted_conditions(clean_summary, shifted_summaries)
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    exploratory_dir = output_dir / "exploratory"
    exploratory_dir.mkdir(parents=True, exist_ok=True)

    ckpt_d: dict | None = None
    if ckpt_file is not None and ckpt_file.is_file():
        try:
            loaded_ckpt = load_checkpoint(ckpt_file, map_location="cpu")
            if isinstance(loaded_ckpt, dict):
                ckpt_d = loaded_ckpt
        except Exception:
            ckpt_d = None

    cms_sub_json_path: Path | None = None
    cms_report: dict | None = None
    cms_summary_meta: dict[str, object] = {}

    ckpt_sha = str(
        cached_bundle.get(
            "checkpoint_sha256",
            file_sha256(ckpt_file) if ckpt_file and ckpt_file.is_file() else "",
        )
    )
    ckpt_epoch = 0
    full_eval_json = output_dir / f"evaluation_{split}.json"
    if full_eval_json.is_file():
        try:
            full_ev_d = json.loads(full_eval_json.read_text(encoding="utf-8"))
            ckpt_epoch = int((full_ev_d.get("checkpoint") or {}).get("epoch", 0))
        except Exception:
            ckpt_epoch = 0
    if ckpt_epoch == 0 and isinstance(ckpt_d, Mapping):
        try:
            ckpt_epoch = int(ckpt_d.get("epoch", 0) or 0)
        except Exception:
            ckpt_epoch = 0

    eval_result: dict[str, object] = {
        "status": "evaluation_completed",
        "dataset": dataset,
        "stage": stage,
        "exp_tag": resolved_exp_tag,
        "word_exp_tag": resolved_word_exp_tag,
        "stage_display_name": STAGE_DISPLAY_NAMES.get(stage, stage),
        "split": split,
        "seed": seed,
        "common_horizon_delay": int(common_horizon_delay),
        "precision": tc["precision"],
        "window_seconds": eval_win_sec,
        "window_samples": eval_win_samp,
        "window_start_offset_seconds": eval_win_off,
        "subset": subset_spec,
        "checkpoint": {
            "path": str(ckpt_file or (output_dir / "best.ckpt")),
            "sha256": ckpt_sha,
            "epoch": ckpt_epoch,
            "selection": "common_horizon_subset",
        },
        "metrics": metrics,
        "negative_controls": negative_controls_result,
        "output_dir": str(output_dir),
        "subspace_encoded_cache_pt": str(encoded_cache_pt),
        "figures": {},
    }
    if cms_sub_json_path is not None:
        eval_result["exploratory_common_mode_subtraction_json"] = str(
            cms_sub_json_path
        )
        eval_result["exploratory_common_mode_subtraction_summary"] = (
            cms_summary_meta
        )

    sub_out_dir = output_dir / "common_subset"
    sub_out_dir.mkdir(parents=True, exist_ok=True)
    print_word_evaluation_report(
        eval_result,
        verbose=verbose,
        output_dir=sub_out_dir,
        print_to_stdout=print_report,
    )
    eval_result["exported_files"] = export_word_evaluation_tables_and_markdown(
        eval_result, sub_out_dir
    )
    eval_json_path = output_dir / f"evaluation_common_subset_{split}.json"
    if promote_to_primary:
        eval_result["promoted_common_subset_to_primary"] = True
        if cms_sub_json_path is not None and isinstance(cms_report, Mapping):
            primary_cms_json = (
                exploratory_dir / f"common_mode_subtraction_{split}.json"
            )
            save_json(primary_cms_json, cms_report)
            eval_result["exploratory_common_mode_subtraction_json"] = str(
                primary_cms_json
            )
        print_word_evaluation_report(
            eval_result,
            verbose=verbose,
            output_dir=output_dir,
            print_to_stdout=False,
        )
        eval_result["exported_files"] = (
            export_word_evaluation_tables_and_markdown(eval_result, output_dir)
        )
        if plot:
            figs = plot_evaluation_figures(eval_result, output_dir / "figures")
            eval_result["figures"].update(figs)
        primary_eval_json = output_dir / f"evaluation_{split}.json"
        eval_result["evaluation_json"] = str(primary_eval_json)
        save_json(eval_json_path, eval_result)
        save_json(primary_eval_json, eval_result)
    else:
        save_json(eval_json_path, eval_result)
        eval_result["evaluation_json"] = str(eval_json_path)
    return eval_result


def _resolve_eval_subdir(eval_subdir: str | None) -> str:
    """规范化评估产物隔离子目录名（为空表示沿用原训练目录）。"""
    raw = (
        str(eval_subdir).strip()
        if eval_subdir is not None
        else os.environ.get("BRAINDECODING_EVAL_SUBDIR", "").strip()
    )
    if raw.lower() in {"", "none", "null", "default"}:
        return ""
    cleaned = raw.strip("/\\")
    if ".." in Path(cleaned).parts:
        raise ValueError(f"非法的 eval_subdir（禁止包含 '..'）：{eval_subdir!r}")
    return cleaned


def _apply_eval_subdir_to_seed_dir(
    seed_dir: Path,
    eval_subdir: str | None,
    *,
    stage: str,
) -> Path:
    """只读保留原目录检查点，将单种子评估输出重定向至 <exp_root>/<eval_subdir>/<stage>/<subset>/<seed>。"""
    sub = _resolve_eval_subdir(eval_subdir)
    if not sub:
        return seed_dir
    resolved = seed_dir.resolve()
    if len(resolved.parts) >= 3 and resolved.parent.parent.name == stage:
        return (
            resolved.parent.parent.parent
            / sub
            / stage
            / resolved.parent.name
            / resolved.name
        )
    return resolved / sub


def _is_encoded_cache_compatible(
    encoded_cache_pt: Path,
    *,
    control_spec: Mapping[str, object],
    ckpt_file: Path | None,
) -> bool:
    """检查已有 subspace_encoded_cache_{split}.pt 是否与当前移位模式、移位集合、检查点及 mu0 要求一致。"""
    if not encoded_cache_pt.is_file():
        return False
    try:
        bundle = torch.load(
            encoded_cache_pt, map_location="cpu", weights_only=False
        )
    except Exception:
        return False
    if not isinstance(bundle, Mapping):
        return False

    cached_ckpt_sha = str(bundle.get("checkpoint_sha256", "") or "").strip()
    if ckpt_file is not None and ckpt_file.is_file() and cached_ckpt_sha:
        try:
            if cached_ckpt_sha != file_sha256(ckpt_file):
                return False
        except Exception:
            return False

    cached_shifts = [
        round(float(x), 4) for x in (bundle.get("shift_seconds") or ())
    ]
    req_shifts = [
        round(float(x), 4) for x in (control_spec.get("shift_seconds") or ())
    ]
    if cached_shifts != req_shifts:
        return False

    req_mode = str(control_spec.get("shift_mode", "circular")).strip().lower()
    raw_cached_mode = bundle.get("shift_mode")
    if req_mode == "circular":
        if str(raw_cached_mode or "circular").strip().lower() != "circular":
            return False
    else:
        if str(raw_cached_mode or "").strip().lower() != req_mode:
            return False
        cached_preroll = bundle.get("preroll_seconds")
        req_preroll = float(control_spec.get("preroll_seconds", 45.0))
        if (
            cached_preroll is not None
            and abs(float(cached_preroll) - req_preroll) >= 1e-4
        ):
            return False

    cms_spec = control_spec.get("common_mode_subtraction") or {}
    if bool(cms_spec.get("enabled", False)) and str(
        cms_spec.get("mode", "loo")
    ).strip().lower() in {"causal_ema", "pos_causal_ema"}:
        if bundle.get("mu0_payload") is None:
            has_ckpt_mu0 = False
            if ckpt_file is not None and ckpt_file.is_file():
                try:
                    ckpt_probe = load_checkpoint(ckpt_file, map_location="cpu")
                    has_ckpt_mu0 = (
                        isinstance(ckpt_probe, Mapping)
                        and ckpt_probe.get("causal_cms_mu0") is not None
                    )
                except Exception:
                    has_ckpt_mu0 = False
            if not has_ckpt_mu0:
                return False

    return True
