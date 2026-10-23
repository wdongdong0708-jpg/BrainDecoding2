"""序列解码协议（Stage 3 / S3）统一命令行执行入口。

支持四步解耦或一键全流程执行：
1. ``export``：提取 S0 / K0 / K1 / K2 / F 在 Clean (0s) 与 Shifted (-75s) 下的 [N, |V|] 神经词表得分矩阵并落盘为 ``.pt``；
2. ``calibrate``：在验证集（val）上按状态独立网格搜索温度 T_r 与语言权重 lambda_r，强制校验 lambda_K0 >= lambda_K1 >= lambda_K2；
3. ``decode``：采用方案一（K2 单遍定稿主干 + K1/K0 轻量草稿延伸）与 KV-Cache 增量批量打分执行序列解码；
4. ``evaluate``：计算 6 维语义指标（BLEU-1, WER, CER, Word Acc [实义/功能词], BERTScore）并扫描 C-gated 门控阈值绘制帕累托曲线；
5. ``all``：按顺序串联执行上述四步。
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
from typing import Any

import numpy as np
import torch

from braindecoding.config import (
    ensure_default_environment,
    load_yaml_with_extends,
    project_path,
    resolve_config_path,
    results_root,
)


DEFAULT_CONFIG_PATH = "configs/decoding/sequence_decoding_protocol.yaml"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="decode_sentences",
        description=__doc__,
    )
    parser._negative_number_matcher = re.compile(
        r"^-\d+(?:\.\d+)?(?:,\s*[+-]?\d+(?:\.\d+)?)*$"
    )
    parser.add_argument(
        "--step",
        choices=("export", "calibrate", "decode", "evaluate", "all"),
        default="all",
        help="要执行的流水线步骤（默认：all）",
    )
    parser.add_argument(
        "--config",
        "-c",
        default=DEFAULT_CONFIG_PATH,
        help=f"S3 序列解码协议 YAML 配置文件路径（默认：{DEFAULT_CONFIG_PATH}）",
    )
    parser.add_argument(
        "--dataset",
        choices=("pallier2025", "chineseeeg2_littleprince"),
        default="pallier2025",
        help="目标数据集（默认：pallier2025）",
    )
    parser.add_argument(
        "--split",
        choices=("val", "test"),
        default="test",
        help="解码与评估目标划分（默认：test；校准步骤固定使用 val）",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="神经模型随机种子（默认：0）",
    )
    parser.add_argument(
        "--subject",
        default="all",
        help="受试者选择：'all' 或单个受试者 ID（如 sub-01）",
    )
    parser.add_argument(
        "--num-subjects",
        type=int,
        default=None,
        help="按标准顺序选取前 N 个有效受试者",
    )
    parser.add_argument(
        "--subjects",
        default=None,
        help="逗号分隔的受试者列表（如 sub-01,sub-02）",
    )
    parser.add_argument(
        "--exp-tag",
        default=None,
        help="可选神经训练实验标签（默认读取环境变量 BRAINDECODING_EXP_TAG）",
    )
    parser.add_argument(
        "--word-exp-tag",
        default=None,
        help="可选阶段 1 (word) 实验标签（默认读取环境变量 BRAINDECODING_WORD_EXP_TAG）",
    )
    parser.add_argument(
        "--window-seconds",
        type=float,
        default=None,
        help="可选覆写单字神经信号时间窗长度（秒，默认沿用检查点）",
    )
    parser.add_argument(
        "--window-start-offset-seconds",
        type=float,
        default=None,
        help="可选覆写相对词起始时刻的窗口起点偏移（秒，默认沿用检查点）",
    )
    parser.add_argument(
        "--language-model",
        default=None,
        help="可选覆写因果语言模型名称或本地路径（默认从 YAML 读取 Qwen/Qwen3-8B-Base）",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="可选覆写神经候选词截断数量 K（默认：20）",
    )
    parser.add_argument(
        "--beam-width",
        type=int,
        default=None,
        help="可选覆写定稿主干束宽 B（默认：5）",
    )
    parser.add_argument(
        "--temperature",
        default=None,
        help="可选覆写校准温度网格（默认：0.05,0.07）",
    )
    parser.add_argument(
        "--cms-eta",
        type=float,
        default=None,
        help="可选覆写历史因果共模扣除系数 eta（默认：0.0）",
    )
    parser.add_argument(
        "--control-shift-seconds",
        type=float,
        default=None,
        help="可选覆写负对照历史错位秒数（默认：-75.0）",
    )
    parser.add_argument(
        "--k-observations",
        default=None,
        help="可选覆写跨被试聚合观测数列表（默认：1,2,5,10）",
    )
    parser.add_argument(
        "--vocab-sizes",
        default=None,
        help="可选覆写多尺度候选词表规模列表（默认：50,100,250,500,1000,2020；设为空或 0 可跳过多尺度扫描）",
    )
    parser.add_argument(
        "--core-vocab-size",
        type=int,
        default=None,
        help="可选覆写对标 SimpleB2T/Jayalath 的高覆盖核心词表规模（默认：250）",
    )
    parser.add_argument(
        "--max-segments",
        type=int,
        default=None,
        help="可选限制最大解码段落数量（用于快速联调冒烟）",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="可选显式指定输出根目录",
    )
    parser.add_argument(
        "--export-path",
        default=None,
        help="可选显式指定神经词表得分 .pt 文件路径",
    )
    parser.add_argument(
        "--calibration-path",
        default=None,
        help="可选显式指定状态自适应超参校准 JSON 路径",
    )
    parser.add_argument(
        "--decoding-path",
        default=None,
        help="可选显式指定解码输出 JSON 路径",
    )
    parser.add_argument(
        "--no-bertscore",
        action="store_true",
        help="禁用预计算词向量直接回退至字符 n-gram 计算 BERTScore",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="禁用可视化图表生成",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="禁用 tqdm 实时进度条",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="强制重跑全部步骤，忽略已保存的匹配缓存与中间解码产物",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="计算设备（默认自动检测 cuda / cpu）",
    )
    return parser


def resolve_default_paths(
    dataset: str,
    split: str,
    seed: int,
    *,
    output_dir: str | Path | None = None,
    exp_tag: str | None = None,
) -> dict[str, Path]:
    if output_dir is not None:
        base_dir = Path(output_dir).resolve()
    else:
        root = results_root() / dataset
        if exp_tag and str(exp_tag).strip().lower() not in {"", "default", "root", "none"}:
            root = root / str(exp_tag).strip()
        base_dir = (root / "sequence_decoding" / f"seed-{int(seed):03d}").resolve()
    return {
        "base_dir": base_dir,
        "val_export_path": base_dir / "neural_vocabulary_scores_val.pt",
        "split_export_path": base_dir / f"neural_vocabulary_scores_{split}.pt",
        "calibration_path": base_dir / "state_hyperparameter_calibration.json",
        "decoding_path": base_dir / f"sequence_decoding_{split}.json",
        "evaluation_dir": base_dir / f"evaluation_{split}",
    }


def _load_or_init_language_scorer(
    protocol_cfg: dict[str, Any],
    *,
    dataset: str,
    vocabulary: list[str],
    language_model_override: str | None = None,
    device: str | None = None,
):
    from braindecoding.decoding.language_model_scoring import (
        CachedVocabularyLanguageScorer,
        PROMPTS,
        load_causal_language_model,
    )

    lm_cfg = dict(protocol_cfg.get("language_model") or {})
    model_name = (
        language_model_override
        or lm_cfg.get("model_name_or_path")
        or lm_cfg.get("model_name")
        or "Qwen/Qwen3-8B-Base"
    )
    precision = str(lm_cfg.get("precision") or lm_cfg.get("dtype") or "bf16-mixed")
    local_files_only = bool(lm_cfg.get("local_files_only", False))
    language = "fr" if dataset == "pallier2025" else "zh"
    prompts = dict(lm_cfg.get("prompts") or PROMPTS)
    prompt_body = str(prompts.get(language, PROMPTS[language])).strip()

    model, tokenizer = load_causal_language_model(
        model_name,
        device=device or "auto",
        precision=precision,
        local_files_only=local_files_only,
    )
    return CachedVocabularyLanguageScorer(
        model=model,
        tokenizer=tokenizer,
        language=language,
        vocabulary=vocabulary,
        settings=lm_cfg,
        prompt_body=prompt_body,
    )


def run_sequence_decoding_pipeline(args: argparse.Namespace) -> dict[str, Any]:
    ensure_default_environment()
    cfg_path = resolve_config_path(project_path(args.config))
    protocol_cfg = load_yaml_with_extends(cfg_path) if cfg_path.is_file() else {}

    dec_cfg = dict(protocol_cfg.get("decoding") or {})
    cal_cfg = dict(protocol_cfg.get("calibration") or {})
    ctrl_cfg = dict(protocol_cfg.get("control") or protocol_cfg.get("controls") or {})
    cms_cfg = dict(ctrl_cfg.get("common_mode_subtraction") or {})
    shift_ctrl_cfg = dict(ctrl_cfg.get("shifted_negative_control") or {})
    eval_cfg = dict(protocol_cfg.get("evaluation") or {})

    exp_tag = (
        args.exp_tag
        if getattr(args, "exp_tag", None) is not None
        else (os.environ.get("BRAINDECODING_EXP_TAG", "").strip() or None)
    )
    word_exp_tag = (
        args.word_exp_tag
        if getattr(args, "word_exp_tag", None) is not None
        else (os.environ.get("BRAINDECODING_WORD_EXP_TAG", "").strip() or None)
    )

    top_k = int(
        args.top_k
        if args.top_k is not None
        else dec_cfg.get("candidate_top_k", dec_cfg.get("top_k_candidates", 20))
    )
    beam_width = int(
        args.beam_width if args.beam_width is not None else dec_cfg.get("beam_width", 5)
    )
    segment_batch_size = int(dec_cfg.get("segment_batch_size", 32))
    cms_eta = float(
        args.cms_eta
        if args.cms_eta is not None
        else cms_cfg.get("subtraction_eta", ctrl_cfg.get("common_mode_subtraction_eta", 0.0))
    )
    cms_shifts = tuple(
        float(s)
        for s in cms_cfg.get(
            "background_shift_seconds",
            ctrl_cfg.get(
                "historical_common_mode_shifts_seconds",
                [-15.0, -30.0, -45.0, -60.0, -90.0, -120.0],
            ),
        )
    )
    control_shift = float(
        args.control_shift_seconds
        if args.control_shift_seconds is not None
        else shift_ctrl_cfg.get(
            "control_shift_seconds",
            ctrl_cfg.get("primary_negative_control_shift_seconds", -75.0),
        )
    )
    raw_k_obs = getattr(args, "k_observations", None)
    if raw_k_obs is not None:
        if isinstance(raw_k_obs, (list, tuple)):
            k_observations = tuple(int(x) for x in raw_k_obs)
        else:
            k_observations = tuple(
                int(x.strip())
                for x in str(raw_k_obs).split(",")
                if x.strip()
            )
    else:
        k_observations = tuple(
            int(x)
            for x in eval_cfg.get("multi_subject_observations", [1, 2, 5, 10])
        )
    core_vocab_size = int(
        args.core_vocab_size
        if getattr(args, "core_vocab_size", None) is not None
        else eval_cfg.get("core_vocabulary_size", 250)
    )
    raw_vocab_sizes = getattr(args, "vocab_sizes", None)
    if raw_vocab_sizes is not None:
        if isinstance(raw_vocab_sizes, (list, tuple)):
            vocab_sizes = tuple(int(x) for x in raw_vocab_sizes if int(x) > 0)
        else:
            s_val = str(raw_vocab_sizes).strip().lower()
            if s_val in ("", "none", "0", "false", "off"):
                vocab_sizes = ()
            else:
                vocab_sizes = tuple(
                    int(x.strip())
                    for x in str(raw_vocab_sizes).split(",")
                    if x.strip() and int(x.strip()) > 0
                )
    else:
        vocab_sizes = tuple(
            int(x)
            for x in eval_cfg.get(
                "vocabulary_scaling_sizes", [50, 100, 250, 500, 1000, 2020]
            )
            if int(x) > 0
        )
    window_seconds = getattr(args, "window_seconds", None)
    window_start_offset_seconds = getattr(args, "window_start_offset_seconds", None)
    show_progress = not bool(getattr(args, "no_progress", False))
    plot_enabled = not bool(getattr(args, "no_plot", False))

    paths = resolve_default_paths(
        args.dataset,
        args.split,
        args.seed,
        output_dir=args.output_dir,
        exp_tag=exp_tag,
    )
    export_path = (
        Path(args.export_path).resolve()
        if args.export_path
        else paths["split_export_path"]
    )
    val_export_path = paths["val_export_path"]
    calibration_path = (
        Path(args.calibration_path).resolve()
        if args.calibration_path
        else paths["calibration_path"]
    )
    decoding_path = (
        Path(args.decoding_path).resolve()
        if args.decoding_path
        else paths["decoding_path"]
    )

    results_summary: dict[str, Any] = {
        "step": args.step,
        "dataset": args.dataset,
        "split": args.split,
        "seed": int(args.seed),
        "exp_tag": exp_tag,
        "cms_eta": cms_eta,
    }

    from braindecoding.decoding.neural_score_exporter import (
        export_neural_vocabulary_scores,
        load_neural_vocabulary_scores,
    )
    from braindecoding.decoding.rolling_beam_search import (
        decode_dataset_segments,
        run_multi_subject_observation_sweep,
        run_vocabulary_scaling_sweep,
    )
    from braindecoding.decoding.state_hyperparameter_calibrator import (
        calibrate_state_hyperparameters,
    )
    from braindecoding.evaluation.sequence_decoding_evaluation import (
        DEFAULT_TAU_GATE_GRID,
        _backfill_intermediate_and_shifted_bundles,
        evaluate_and_export_sequence_decoding,
    )

    force_rerun = bool(getattr(args, "force", False))

    def _run_export_for_split(target_split: str, target_out: Path) -> Path:
        return export_neural_vocabulary_scores(
            args.dataset,
            split=target_split,
            seed=args.seed,
            subject=args.subject,
            num_subjects=args.num_subjects,
            subjects=args.subjects,
            exp_tag=exp_tag,
            word_exp_tag=word_exp_tag,
            window_seconds=window_seconds,
            window_start_offset_seconds=window_start_offset_seconds,
            background_shift_seconds=cms_shifts,
            control_shift_seconds=control_shift,
            subtraction_eta=cms_eta,
            output_path=target_out,
            device=args.device or "auto",
            show_progress=show_progress,
        )

    def _export_matches_cms_eta(pt_path: Path, expected_eta: float) -> bool:
        if not pt_path.is_file():
            return False
        try:
            payload = load_neural_vocabulary_scores(pt_path)
            cms_meta = dict(payload.get("cms_metadata") or {})
            if "subtraction_eta" not in cms_meta:
                return True
            return abs(float(cms_meta["subtraction_eta"]) - float(expected_eta)) <= 1e-6
        except Exception:
            return False

    if args.step in ("export", "all"):
        if (
            args.step == "all"
            and args.split != "val"
            and (force_rerun or not _export_matches_cms_eta(val_export_path, cms_eta))
        ):
            _run_export_for_split("val", val_export_path)
        if (
            args.step == "all"
            and not force_rerun
            and _export_matches_cms_eta(export_path, cms_eta)
        ):
            out_pt = export_path
        else:
            out_pt = _run_export_for_split(args.split, export_path)
        exported_payload = load_neural_vocabulary_scores(out_pt)
        results_summary["export_path"] = str(out_pt)
        results_summary["vocabulary_size"] = int(exported_payload["vocabulary_size"])

    scorer = None

    if args.step in ("calibrate", "all"):
        raw_temp_override = getattr(args, "temperature", None)
        if raw_temp_override is not None:
            if isinstance(raw_temp_override, (list, tuple)):
                temp_grid = tuple(float(x) for x in raw_temp_override)
            else:
                temp_grid = tuple(
                    float(x.strip())
                    for x in str(raw_temp_override).split(",")
                    if x.strip()
                )
        else:
            temp_grid = tuple(
                float(x)
                for x in cal_cfg.get("temperature_grid", [0.05, 0.07])
            )
        lambda_grid = tuple(
            float(x)
            for x in cal_cfg.get(
                "language_weight_grid",
                [0.00, 0.05, 0.10, 0.15, 0.25, 0.40, 0.60, 1.00],
            )
        )

        reused_calibration: dict[str, Any] | None = None
        if args.step == "all" and not force_rerun and calibration_path.is_file():
            try:
                existing_cal = json.loads(
                    calibration_path.read_text(encoding="utf-8")
                )
                saved_temps = [
                    round(float(t), 6)
                    for t in existing_cal.get("temperature_grid") or []
                ]
                expected_temps = sorted(
                    {round(float(t), 6) for t in temp_grid if float(t) > 0.0}
                )
                saved_eta = existing_cal.get("cms_eta")
                eta_ok = saved_eta is None or (
                    abs(float(saved_eta) - float(cms_eta)) <= 1e-6
                )
                if (
                    saved_temps == expected_temps
                    and eta_ok
                    and isinstance(existing_cal.get("states"), dict)
                    and all(
                        k in existing_cal["states"] for k in ("K0", "K1", "K2")
                    )
                ):
                    reused_calibration = existing_cal
            except Exception:
                reused_calibration = None

        if reused_calibration is not None:
            calibration_res = reused_calibration
        else:
            cal_source_path = (
                val_export_path
                if (args.split != "val" or val_export_path.is_file())
                else export_path
            )
            if force_rerun or not _export_matches_cms_eta(cal_source_path, cms_eta):
                _run_export_for_split("val", cal_source_path)
            val_payload = load_neural_vocabulary_scores(cal_source_path)
            scorer = _load_or_init_language_scorer(
                protocol_cfg,
                dataset=args.dataset,
                vocabulary=list(val_payload["vocabulary"]),
                language_model_override=args.language_model,
                device=args.device,
            )
            calibration_res = calibrate_state_hyperparameters(
                val_payload,
                scorer,
                temperature_grid=temp_grid,
                language_weight_grid=lambda_grid,
                beam_width=beam_width,
                candidate_top_k=top_k,
                segment_batch_size=segment_batch_size,
                metric_name=str(cal_cfg.get("metric", "composite")),
                enforce_monotonic_lambda=bool(
                    cal_cfg.get("enforce_monotonic_lambda", True)
                ),
                use_calibration_partition=bool(
                    cal_cfg.get("use_calibration_partition", True)
                ),
                max_calibration_segments=args.max_segments
                or cal_cfg.get("max_calibration_segments", 64),
                output_path=calibration_path,
                show_progress=show_progress,
            )
        results_summary["calibration_path"] = str(calibration_path)
        results_summary["state_hyperparameters"] = calibration_res["states"]

    if args.step in ("decode", "all"):
        if not _export_matches_cms_eta(export_path, cms_eta):
            _run_export_for_split(args.split, export_path)
        split_payload = load_neural_vocabulary_scores(export_path)
        if calibration_path.is_file():
            cal_json = json.loads(calibration_path.read_text(encoding="utf-8"))
            hyperparameters = cal_json
        else:
            hyperparameters = {
                "states": {
                    s: {"temperature": 0.07, "language_weight": 0.15}
                    for s in ("S0", "K0", "K1", "K2", "F")
                }
            }

        decoded_bundles: dict[str, Any] = {}
        multi_subject_bundles: dict[str, Any] = {}
        vocabulary_scaling_bundles: dict[str, Any] = {}
        available_clean = split_payload.get("clean_scores", {})
        ckpt_prov = dict(split_payload.get("checkpoint_provenance") or {})
        f_mode = (
            "F_local_fallback"
            if ckpt_prov.get("context")
            in (None, "", "fallback_from_revision_mature")
            else "F_full_context"
        )
        total_split_segs = len(split_payload.get("segments") or [])
        expected_seg_count = (
            min(total_split_segs, int(args.max_segments))
            if args.max_segments is not None
            else total_split_segs
        )
        target_states_hp = hyperparameters.get("states", hyperparameters)

        if not force_rerun and decoding_path.is_file():
            try:
                existing_dec = json.loads(
                    decoding_path.read_text(encoding="utf-8")
                )
                saved_hp = existing_dec.get("state_hyperparameters") or {}
                hp_matches = all(
                    isinstance(saved_hp.get(st), dict)
                    and isinstance(target_states_hp.get(st), dict)
                    and abs(
                        float(saved_hp[st].get("temperature", -1.0))
                        - float(target_states_hp[st].get("temperature", -2.0))
                    )
                    <= 1e-6
                    and abs(
                        float(saved_hp[st].get("language_weight", -1.0))
                        - float(target_states_hp[st].get("language_weight", -2.0))
                    )
                    <= 1e-6
                    for st in ("K0", "K1", "K2")
                    if st in target_states_hp
                )
                if (
                    existing_dec.get("dataset") == args.dataset
                    and existing_dec.get("split") == args.split
                    and int(existing_dec.get("seed", -1)) == int(args.seed)
                    and hp_matches
                ):
                    for cond_k, cond_b in (
                        existing_dec.get("decoded_bundles") or {}
                    ).items():
                        if (
                            isinstance(cond_b, dict)
                            and int(cond_b.get("segment_count", -1))
                            == expected_seg_count
                        ):
                            decoded_bundles[cond_k] = cond_b
                    if isinstance(existing_dec.get("multi_subject_bundles"), dict):
                        multi_subject_bundles = dict(
                            existing_dec["multi_subject_bundles"]
                        )
                    for scale_k, scale_b in (
                        existing_dec.get("vocabulary_scaling_bundles") or {}
                    ).items():
                        if not isinstance(scale_b, dict):
                            continue
                        conds_b = scale_b.get("conditions")
                        s3_b = (
                            conds_b.get("S3_clean")
                            if isinstance(conds_b, dict)
                            else None
                        )
                        if (
                            isinstance(s3_b, dict)
                            and int(s3_b.get("segment_count", -1))
                            == expected_seg_count
                        ):
                            vocabulary_scaling_bundles[str(scale_k)] = scale_b
            except Exception:
                decoded_bundles = {}
                multi_subject_bundles = {}
                vocabulary_scaling_bundles = {}

        def _ensure_scorer():
            nonlocal scorer
            if scorer is None:
                scorer = _load_or_init_language_scorer(
                    protocol_cfg,
                    dataset=args.dataset,
                    vocabulary=list(split_payload["vocabulary"]),
                    language_model_override=args.language_model,
                    device=args.device,
                )
            return scorer

        def _persist_decoding_bundle() -> None:
            _backfill_intermediate_and_shifted_bundles(decoded_bundles)
            bundle_obj = {
                "schema_version": "s3-decoding-bundle/v1",
                "dataset": args.dataset,
                "split": args.split,
                "seed": int(args.seed),
                "language": "fr" if args.dataset == "pallier2025" else "zh",
                "checkpoint_provenance": ckpt_prov,
                "f_condition_mode": f_mode,
                "state_hyperparameters": target_states_hp,
                "decoded_bundles": decoded_bundles,
                "multi_subject_bundles": multi_subject_bundles,
                "vocabulary_scaling_bundles": vocabulary_scaling_bundles,
            }
            decoding_path.parent.mkdir(parents=True, exist_ok=True)
            decoding_path.write_text(
                json.dumps(bundle_obj, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )

        for baseline_mode in ("LM-Only", "S0", "K0"):
            if baseline_mode != "LM-Only" and baseline_mode not in available_clean:
                continue
            if baseline_mode in decoded_bundles:
                continue
            decoded_bundles[baseline_mode] = decode_dataset_segments(
                split_payload,
                _ensure_scorer(),
                hyperparameters,
                condition=baseline_mode,
                alignment="clean",
                beam_width=beam_width,
                candidate_top_k=top_k,
                segment_batch_size=segment_batch_size,
                max_segments=args.max_segments,
                show_progress=show_progress,
                progress_desc=f"S3 Decode [{baseline_mode} | {args.split}]",
            )
            _persist_decoding_bundle()

        if "S3_clean" not in decoded_bundles:
            decoded_bundles["S3_clean"] = decode_dataset_segments(
                split_payload,
                _ensure_scorer(),
                hyperparameters,
                condition="S3",
                alignment="clean",
                beam_width=beam_width,
                candidate_top_k=top_k,
                segment_batch_size=segment_batch_size,
                max_segments=args.max_segments,
                show_progress=show_progress,
                progress_desc=f"S3 Decode [S3_clean | {args.split}]",
            )
            _persist_decoding_bundle()

        if "F" in available_clean and "F" not in decoded_bundles:
            decoded_bundles["F"] = decode_dataset_segments(
                split_payload,
                _ensure_scorer(),
                hyperparameters,
                condition="F",
                alignment="clean",
                beam_width=beam_width,
                candidate_top_k=top_k,
                segment_batch_size=segment_batch_size,
                max_segments=args.max_segments,
                show_progress=show_progress,
                progress_desc=f"S3 Decode [F ({f_mode}) | {args.split}]",
            )
            _persist_decoding_bundle()

        if "S3_shifted_neg75s" not in decoded_bundles:
            decoded_bundles["S3_shifted_neg75s"] = decode_dataset_segments(
                split_payload,
                _ensure_scorer(),
                hyperparameters,
                condition="S3",
                alignment="shifted",
                beam_width=beam_width,
                candidate_top_k=top_k,
                segment_batch_size=segment_batch_size,
                max_segments=args.max_segments,
                show_progress=show_progress,
                progress_desc=f"S3 Decode [S3_shifted_-75s | {args.split}]",
            )
            _persist_decoding_bundle()

        unique_subjects = {
            str(seg.get("subject_id", "default"))
            for seg in split_payload.get("segments") or []
        }
        expected_k_keys = [
            f"k={int(k)}"
            for k in sorted({int(x) for x in k_observations if int(x) >= 1})
            if int(k) <= len(unique_subjects)
        ]
        if (
            len(k_observations) >= 2
            and len(unique_subjects) > 1
            and not all(k_key in multi_subject_bundles for k_key in expected_k_keys)
        ):
            multi_subject_bundles = run_multi_subject_observation_sweep(
                split_payload,
                _ensure_scorer(),
                hyperparameters,
                k_observations=k_observations,
                condition="S3",
                alignment="clean",
                single_subject_s3_bundle=decoded_bundles.get("S3_clean"),
                beam_width=beam_width,
                candidate_top_k=top_k,
                segment_batch_size=segment_batch_size,
                max_segments=args.max_segments,
                show_progress=show_progress,
            )
            _persist_decoding_bundle()

        if len(vocab_sizes) > 0:
            full_vocab_len = len(split_payload.get("vocabulary") or [])
            effective_scales = sorted(
                {
                    min(int(sz), full_vocab_len)
                    for sz in vocab_sizes
                    if int(sz) >= 1
                }
            )
            needs_truncated_decode = any(
                int(n) < full_vocab_len
                and (
                    f"V={int(n)}" not in vocabulary_scaling_bundles
                    or "S3_clean"
                    not in (
                        vocabulary_scaling_bundles[f"V={int(n)}"].get("conditions")
                        or {}
                    )
                    or "S3_shifted_neg75s"
                    not in (
                        vocabulary_scaling_bundles[f"V={int(n)}"].get("conditions")
                        or {}
                    )
                )
                for n in effective_scales
            )
            active_scorer = _ensure_scorer() if needs_truncated_decode else scorer

            def _on_scale_completed(
                scale_key: str, scale_bundle: dict[str, Any]
            ) -> None:
                vocabulary_scaling_bundles[scale_key] = scale_bundle
                _persist_decoding_bundle()

            vocabulary_scaling_bundles = run_vocabulary_scaling_sweep(
                split_payload,
                active_scorer,
                hyperparameters,
                vocab_sizes=vocab_sizes,
                core_vocab_size=core_vocab_size,
                full_vocab_bundles=decoded_bundles,
                existing_scaling_bundles=vocabulary_scaling_bundles,
                beam_width=beam_width,
                candidate_top_k=top_k,
                segment_batch_size=segment_batch_size,
                max_segments=args.max_segments,
                show_progress=show_progress,
                on_scale_completed=_on_scale_completed,
            )

        _persist_decoding_bundle()
        results_summary["decoding_path"] = str(decoding_path)
        results_summary["checkpoint_provenance"] = ckpt_prov
        results_summary["f_condition_mode"] = f_mode

    if args.step in ("evaluate", "all"):
        bundle = json.loads(decoding_path.read_text(encoding="utf-8"))
        language = str(
            bundle.get("language", "fr" if args.dataset == "pallier2025" else "zh")
        )
        decoded_bundles_map = dict(bundle.get("decoded_bundles") or {})
        multi_subject_bundles_map = dict(bundle.get("multi_subject_bundles") or {})
        vocabulary_scaling_bundles_map = dict(
            bundle.get("vocabulary_scaling_bundles") or {}
        )

        word_embedding_map = None
        if not bool(args.no_bertscore) and export_path.is_file():
            split_payload = load_neural_vocabulary_scores(export_path)
            vocab_list = list(split_payload.get("vocabulary") or [])
            vocab_embs = split_payload.get("vocabulary_embeddings")
            if vocab_embs is not None and len(vocab_list) == len(vocab_embs):
                emb_arr = (
                    vocab_embs.detach().cpu().numpy()
                    if isinstance(vocab_embs, torch.Tensor)
                    else np.asarray(vocab_embs, dtype=np.float32)
                )
                word_embedding_map = {
                    w: emb_arr[idx] for idx, w in enumerate(vocab_list)
                }

        cg_cfg = dict(eval_cfg.get("confidence_gate") or {})
        if "threshold_sweep" in cg_cfg:
            tau_grid = tuple(float(t) for t in cg_cfg["threshold_sweep"])
        elif "tau_min" in cg_cfg and "tau_max" in cg_cfg:
            t_min = float(cg_cfg.get("tau_min", 0.0))
            t_max = float(cg_cfg.get("tau_max", 2.0))
            t_step = float(cg_cfg.get("tau_step", 0.1))
            count = int(round((t_max - t_min) / max(t_step, 1e-6))) + 1
            tau_grid = tuple(round(t_min + i * t_step, 4) for i in range(count))
        else:
            tau_grid = DEFAULT_TAU_GATE_GRID

        eval_report = evaluate_and_export_sequence_decoding(
            decoded_bundles_map,
            dataset=args.dataset,
            language=language,
            split=args.split,
            output_dir=paths["evaluation_dir"],
            word_embedding_map=word_embedding_map,
            tau_grid=tau_grid,
            multi_subject_bundles=multi_subject_bundles_map,
            vocabulary_scaling_bundles=vocabulary_scaling_bundles_map,
            core_vocab_size=core_vocab_size,
            plot=plot_enabled,
        )
        results_summary["evaluation_dir"] = str(paths["evaluation_dir"])
        results_summary["conditions"] = eval_report["conditions"]
        results_summary["multi_subject_scaling"] = eval_report.get(
            "multi_subject_scaling", {}
        )
        results_summary["core_vocabulary_benchmark"] = eval_report.get(
            "core_vocabulary_benchmark", {}
        )
        results_summary["full_vocab_core_subsets"] = eval_report.get(
            "full_vocab_core_subsets", {}
        )
        results_summary["vocabulary_scaling"] = eval_report.get(
            "vocabulary_scaling", {}
        )
        results_summary["figures"] = eval_report["figures"]

    return results_summary


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    summary = run_sequence_decoding_pipeline(args)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
