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
import copy
import json
import os
from pathlib import Path
import re
from typing import Any

import numpy as np
import torch

from braindecoding.cli import _parse_seed_argument
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
        help=(
            f"S3 序列解码协议 YAML 或神经训练阶段 YAML 配置文件路径（默认：{DEFAULT_CONFIG_PATH}；"
            "若传入如 configs/training/siamese_debiased/pallier2025_siamese_debiased_prefix_revision.yaml 的训练配置，"
            "将自动加载默认 S3 协议并提取 dataset、revision_stage 与 window_seconds）"
        ),
    )
    parser.add_argument(
        "--training-config",
        default=None,
        help="可选显式指定神经编码器训练阶段 YAML 配置文件路径（如 configs/training/siamese_debiased/pallier2025_siamese_debiased_prefix_revision.yaml）",
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
        "--seeds",
        dest="seed",
        type=_parse_seed_argument,
        default=0,
        help="神经模型随机种子：支持单种子（如 0）或逗号分隔的多种子列表（如 0,1,2；默认：0）",
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
        "--context-exp-tag",
        default=None,
        help="可选阶段 2.1 (context) 实验标签（默认读取环境变量 BRAINDECODING_CONTEXT_EXP_TAG；填 off/disable 可禁用 Full-Context）",
    )
    parser.add_argument(
        "--revision-exp-tag",
        default=None,
        help="可选阶段 2.2/2.3 (revision/prefix_revision) 实验标签（默认跟随 --exp-tag）",
    )
    parser.add_argument(
        "--word-config",
        default=None,
        help="可选阶段 1 (word) 训练 YAML 配置文件路径",
    )
    parser.add_argument(
        "--context-config",
        default=None,
        help="可选阶段 2.1 (context) 训练 YAML 配置文件路径",
    )
    parser.add_argument(
        "--revision-config",
        default=None,
        help="可选阶段 2.2/2.3 (revision/prefix_revision) 训练 YAML 配置文件路径",
    )
    parser.add_argument(
        "--word-checkpoint",
        default=None,
        help="可选显式指定阶段 1 (word) 检查点文件路径",
    )
    parser.add_argument(
        "--context-checkpoint",
        default=None,
        help="可选显式指定阶段 2.1 (context) 检查点文件路径",
    )
    parser.add_argument(
        "--revision-checkpoint",
        default=None,
        help="可选显式指定阶段 2.2/2.3 (revision/prefix_revision) 检查点文件路径",
    )
    parser.add_argument(
        "--revision-stage",
        choices=("auto", "revision", "prefix_revision"),
        default="auto",
        help="可选指定 K0/K1/K2 神经表征来源阶段：auto（默认优先 revision 并自动回退 prefix_revision）、revision (Stage 2.2) 或 prefix_revision (Stage 2.3)",
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
        "--json",
        dest="json_output",
        action="store_true",
        help="输出原始 JSON 而非格式化对比表格",
    )
    parser.add_argument(
        "--shared-language-weight",
        action="store_true",
        default=None,
        help="校准时令 K0/K1/K2 共享同一语言模型权重 λ_r >= 0.05（默认按状态自适应正下界单调递减 λ_K0 >= λ_K1 >= λ_K2 >= 0.05）",
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
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="在终端打印完整宽表拆解报告（默认输出单屏 <= 86 列核心执行摘要）",
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
    raw_seed = getattr(args, "seed", 0)
    if isinstance(raw_seed, (list, tuple)):
        seed_list = [int(s) for s in raw_seed]
    elif isinstance(raw_seed, str) and ("," in raw_seed or " " in raw_seed):
        parsed_s = _parse_seed_argument(raw_seed)
        seed_list = parsed_s if isinstance(parsed_s, list) else [int(parsed_s)]
    else:
        seed_list = [int(raw_seed)]
    if len(seed_list) > 1:
        per_seed_summaries: dict[str, dict[str, Any]] = {}
        shared_scorer_box: dict[str, Any] = getattr(
            args, "_shared_scorer_box", {}
        )
        for idx, seed_val in enumerate(seed_list):
            print(
                f"\n{'#' * 86}\n"
                f" [Multi-Seed S3 Sequence Decoding {idx + 1}/{len(seed_list)}] "
                f"Running seed={seed_val} (all seeds: {seed_list})\n"
                f"{'#' * 86}",
                flush=True,
            )
            seed_args = copy.copy(args)
            seed_args.seed = int(seed_val)
            seed_args.output_dir = None
            seed_args.export_path = None
            seed_args.calibration_path = None
            seed_args.decoding_path = None
            seed_args._shared_scorer_box = shared_scorer_box
            seed_args._in_multi_seed_loop = True
            per_seed_summaries[f"seed-{int(seed_val):03d}"] = (
                run_sequence_decoding_pipeline(seed_args)
            )
        multi_seed_summary: dict[str, Any] = {}
        if args.step in ("evaluate", "all") and per_seed_summaries:
            from braindecoding.evaluation.sequence_decoding_evaluation import (
                summarize_and_export_multi_seed_sequence_decoding,
            )

            first_seed_Summary = next(iter(per_seed_summaries.values()))
            first_eval_dir = first_seed_Summary.get("evaluation_dir")
            seq_root_dir = (
                Path(first_eval_dir).resolve().parent.parent
                if first_eval_dir
                else resolve_default_paths(
                    args.dataset,
                    args.split,
                    seed_list[0],
                    output_dir=args.output_dir,
                    exp_tag=getattr(args, "exp_tag", None),
                )["base_dir"].parent
            )
            multi_seed_summary = summarize_and_export_multi_seed_sequence_decoding(
                seq_root_dir,
                split=args.split,
                dataset=args.dataset,
                exp_tag=first_seed_Summary.get("exp_tag"),
                seed_reports=list(per_seed_summaries.values()),
                plot=not bool(getattr(args, "no_plot", False)),
            )
        return {
            "dataset": args.dataset,
            "split": args.split,
            "seeds": seed_list,
            "per_seed_summaries": per_seed_summaries,
            "multi_seed_summary": multi_seed_summary,
        }
    args.seed = int(seed_list[0])
    default_proto_path = resolve_config_path(project_path(DEFAULT_CONFIG_PATH))
    cfg_path = resolve_config_path(project_path(args.config or DEFAULT_CONFIG_PATH))
    loaded_cfg = load_yaml_with_extends(cfg_path) if cfg_path.is_file() else {}

    training_cfg_arg = getattr(args, "training_config", None)
    training_cfg_path: Path | None = None
    training_yaml: dict[str, Any] | None = None

    is_training_yaml_in_config = (
        isinstance(loaded_cfg, dict)
        and any(k in loaded_cfg for k in ("workflow", "siamese_debiased", "model", "training"))
        and "decoding" not in loaded_cfg
        and "language_model" not in loaded_cfg
    )
    if is_training_yaml_in_config:
        training_cfg_path = cfg_path
        training_yaml = loaded_cfg
        protocol_cfg = (
            load_yaml_with_extends(default_proto_path)
            if default_proto_path.is_file()
            else {}
        )
    else:
        protocol_cfg = loaded_cfg
        if training_cfg_arg is not None and str(training_cfg_arg).strip():
            cand_train_p = resolve_config_path(project_path(training_cfg_arg))
            if cand_train_p.is_file():
                raw_t = load_yaml_with_extends(cand_train_p)
                if isinstance(raw_t, dict):
                    training_cfg_path = cand_train_p
                    training_yaml = raw_t

    effective_rev_stage = str(getattr(args, "revision_stage", "auto") or "auto").strip().lower()
    yaml_exp_tag: str | None = None
    yaml_word_exp_tag: str | None = None
    yaml_win_sec: float | None = None
    yaml_win_off: float | None = None
    if isinstance(training_yaml, dict):
        from braindecoding.training.workflow_runner import (
            _infer_dataset_and_stage_from_yaml,
        )

        inferred_ds, inferred_stage = _infer_dataset_and_stage_from_yaml(training_yaml)
        args.dataset = inferred_ds
        if effective_rev_stage == "auto" and inferred_stage in {"revision", "prefix_revision"}:
            effective_rev_stage = inferred_stage
            args.revision_stage = inferred_stage
        wf_b = training_yaml.get("workflow") if isinstance(training_yaml.get("workflow"), dict) else {}
        exp_b = training_yaml.get("experiment") if isinstance(training_yaml.get("experiment"), dict) else {}
        ds_b = training_yaml.get("dataset") if isinstance(training_yaml.get("dataset"), dict) else {}
        yaml_exp_tag = wf_b.get("exp_tag") or exp_b.get("exp_tag")
        yaml_word_exp_tag = wf_b.get("word_exp_tag") or exp_b.get("word_exp_tag")
        raw_w = wf_b.get("window_seconds") if wf_b.get("window_seconds") is not None else ds_b.get("window_seconds")
        raw_o = (
            wf_b.get("window_start_offset_seconds")
            if wf_b.get("window_start_offset_seconds") is not None
            else ds_b.get("window_start_offset_seconds")
        )
        if raw_w is not None:
            yaml_win_sec = float(raw_w)
        if raw_o is not None:
            yaml_win_off = float(raw_o)

    dec_cfg = dict(protocol_cfg.get("decoding") or {})
    cal_cfg = dict(protocol_cfg.get("calibration") or {})
    ctrl_cfg = dict(protocol_cfg.get("control") or protocol_cfg.get("controls") or {})
    cms_cfg = dict(ctrl_cfg.get("common_mode_subtraction") or {})
    shift_ctrl_cfg = dict(ctrl_cfg.get("shifted_negative_control") or {})
    eval_cfg = dict(protocol_cfg.get("evaluation") or {})

    exp_tag = (
        args.exp_tag
        if getattr(args, "exp_tag", None) is not None
        else (os.environ.get("BRAINDECODING_EXP_TAG", "").strip() or yaml_exp_tag or None)
    )
    word_exp_tag = (
        args.word_exp_tag
        if getattr(args, "word_exp_tag", None) is not None
        else (os.environ.get("BRAINDECODING_WORD_EXP_TAG", "").strip() or yaml_word_exp_tag or None)
    )
    context_exp_tag = (
        getattr(args, "context_exp_tag", None)
        if getattr(args, "context_exp_tag", None) is not None
        else (os.environ.get("BRAINDECODING_CONTEXT_EXP_TAG", "").strip() or None)
    )
    revision_exp_tag = (
        getattr(args, "revision_exp_tag", None)
        if getattr(args, "revision_exp_tag", None) is not None
        else exp_tag
    )
    word_config_arg = getattr(args, "word_config", None)
    context_config_arg = getattr(args, "context_config", None)
    revision_config_arg = getattr(args, "revision_config", None)
    word_ckpt_arg = getattr(args, "word_checkpoint", None)
    context_ckpt_arg = getattr(args, "context_checkpoint", None)
    revision_ckpt_arg = getattr(args, "revision_checkpoint", None)

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
    core_vocab_sizes = tuple(
        int(x)
        for x in eval_cfg.get("core_vocabulary_sizes", [50, 100, 250])
        if int(x) > 0
    )
    multi_subject_vocab_sizes = tuple(
        int(x)
        for x in eval_cfg.get(
            "multi_subject_vocabulary_sizes", [50, 100, 250, 2020]
        )
        if int(x) > 0
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
    window_seconds = (
        getattr(args, "window_seconds", None)
        if getattr(args, "window_seconds", None) is not None
        else yaml_win_sec
    )
    window_start_offset_seconds = (
        getattr(args, "window_start_offset_seconds", None)
        if getattr(args, "window_start_offset_seconds", None) is not None
        else yaml_win_off
    )
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
        _is_fake_provenance,
        _same_checkpoint_path,
        export_neural_vocabulary_scores,
        load_neural_vocabulary_scores,
        resolve_stage_checkpoints,
    )
    from braindecoding.decoding.rolling_beam_search import (
        decode_dataset_segment_jobs,
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
    resolved_ckpts = resolve_stage_checkpoints(
        args.dataset,
        seed=args.seed,
        subject=args.subject,
        num_subjects=args.num_subjects,
        subjects=args.subjects,
        exp_tag=exp_tag,
        word_exp_tag=word_exp_tag,
        context_exp_tag=context_exp_tag,
        revision_exp_tag=revision_exp_tag,
        word_checkpoint=word_ckpt_arg,
        context_checkpoint=context_ckpt_arg,
        revision_checkpoint=revision_ckpt_arg,
        revision_stage=effective_rev_stage,
        config_path=training_cfg_path,
        word_config=word_config_arg,
        context_config=context_config_arg,
        revision_config=revision_config_arg,
        window_seconds=window_seconds,
        window_start_offset_seconds=window_start_offset_seconds,
    )

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
            context_exp_tag=context_exp_tag,
            revision_exp_tag=revision_exp_tag,
            word_checkpoint=word_ckpt_arg,
            context_checkpoint=context_ckpt_arg,
            revision_checkpoint=revision_ckpt_arg,
            revision_stage=effective_rev_stage,
            config_path=training_cfg_path,
            word_config=word_config_arg,
            context_config=context_config_arg,
            revision_config=revision_config_arg,
            window_seconds=window_seconds,
            window_start_offset_seconds=window_start_offset_seconds,
            background_shift_seconds=cms_shifts,
            control_shift_seconds=control_shift,
            subtraction_eta=cms_eta,
            reuse_existing_states=not force_rerun,
            output_path=target_out,
            device=args.device or "auto",
            show_progress=show_progress,
        )

    def _export_matches_cms_eta(pt_path: Path, expected_eta: float) -> bool:
        if not pt_path.is_file():
            return False
        try:
            payload = load_neural_vocabulary_scores(pt_path)
            ckpt_prov = dict(
                payload.get("checkpoint_provenance")
                or payload.get("checkpoints")
                or {}
            )
            clean_sc = dict(payload.get("clean_scores") or {})
            for p_val in (
                ckpt_prov.get("word"),
                ckpt_prov.get("context"),
                ckpt_prov.get("revision"),
            ):
                txt = str(p_val or "").strip()
                if txt.startswith("fallback_from_") or txt.startswith(
                    "full_sentence_transformer_from_"
                ):
                    return False

            saved_rev_stage = ckpt_prov.get("revision_stage")
            if (
                effective_rev_stage in ("revision", "prefix_revision")
                and saved_rev_stage
                and saved_rev_stage != "not_available"
                and saved_rev_stage != effective_rev_stage
            ):
                return False

            cur_rev = resolved_ckpts.get("revision_checkpoint")
            cur_word = resolved_ckpts.get("word_checkpoint")
            cur_ctx = resolved_ckpts.get("context_checkpoint")
            if cur_rev is not None:
                if not all(k in clean_sc for k in ("K0", "K1", "K2")):
                    return False
                if not _same_checkpoint_path(ckpt_prov.get("revision"), cur_rev):
                    return False
            if cur_word is not None:
                if "S0" not in clean_sc:
                    return False
                if not _same_checkpoint_path(ckpt_prov.get("word"), cur_word):
                    return False
            if cur_ctx is not None:
                if "F" not in clean_sc:
                    return False
                if not _same_checkpoint_path(ckpt_prov.get("context"), cur_ctx):
                    return False
            elif (
                (cur_rev is not None or cur_word is not None)
                and "F" in clean_sc
            ):
                return False

            cms_meta = dict(payload.get("cms_metadata") or {})
            if "subtraction_eta" not in cms_meta:
                return True
            return abs(float(cms_meta["subtraction_eta"]) - float(expected_eta)) <= 1e-6
        except Exception:
            return False

    export_was_regenerated = False
    if args.step in ("export", "all"):
        if (
            args.step == "all"
            and not force_rerun
            and _export_matches_cms_eta(export_path, cms_eta)
        ):
            out_pt = export_path
        else:
            out_pt = _run_export_for_split(args.split, export_path)
            export_was_regenerated = True
        exported_payload = load_neural_vocabulary_scores(out_pt)
        results_summary["export_path"] = str(out_pt)
        results_summary["vocabulary_size"] = int(exported_payload["vocabulary_size"])

    shared_scorer_box: dict[str, Any] = getattr(args, "_shared_scorer_box", {})
    scorer = None

    def _get_or_init_shared_scorer(vocab: list[str]):
        nonlocal scorer
        if scorer is not None:
            return scorer
        cached_sc = shared_scorer_box.get("scorer")
        cached_vocab = shared_scorer_box.get("vocabulary")
        if cached_sc is not None and cached_vocab == list(vocab):
            scorer = cached_sc
            return scorer
        scorer = _load_or_init_language_scorer(
            protocol_cfg,
            dataset=args.dataset,
            vocabulary=list(vocab),
            language_model_override=args.language_model,
            device=args.device,
        )
        shared_scorer_box["scorer"] = scorer
        shared_scorer_box["vocabulary"] = list(vocab)
        return scorer

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
                [0.05, 0.10, 0.15, 0.25, 0.40, 0.60],
            )
        )
        min_lam_cfg = float(cal_cfg.get("min_language_weight", 0.05))
        shared_lam_arg = getattr(args, "shared_language_weight", None)
        shared_lam_cfg = (
            bool(shared_lam_arg)
            if shared_lam_arg is not None
            else bool(cal_cfg.get("shared_language_weight", False))
        )

        ref_prov: dict[str, Any] = {}
        required_cal_states: tuple[str, ...] = ()
        if "exported_payload" in locals() and isinstance(exported_payload, dict):
            ref_prov = dict(exported_payload.get("checkpoint_provenance") or {})
            ref_clean = dict(exported_payload.get("clean_scores") or {})
            required_cal_states = tuple(
                st for st in ("S0", "K0", "K1", "K2", "F") if st in ref_clean
            )

        reusable_cal_states: dict[str, dict[str, Any]] = {}
        existing_cal_base: dict[str, Any] | None = None
        if not force_rerun and calibration_path.is_file():
            try:
                existing_cal = json.loads(
                    calibration_path.read_text(encoding="utf-8")
                )
                saved_cal_stage = existing_cal.get("revision_stage")
                cal_stage_ok = (
                    not saved_cal_stage
                    or effective_rev_stage not in ("revision", "prefix_revision")
                    or saved_cal_stage == effective_rev_stage
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
                shared_mode_ok = (
                    bool(existing_cal.get("shared_language_weight", False))
                    == shared_lam_cfg
                )
                saved_states = existing_cal.get("states")
                saved_cal_prov = dict(
                    existing_cal.get("checkpoint_provenance") or {}
                )
                if (
                    saved_temps == expected_temps
                    and eta_ok
                    and shared_mode_ok
                    and isinstance(saved_states, dict)
                ):
                    existing_cal_base = existing_cal
                    cur_w = ref_prov.get("word") or resolved_ckpts.get(
                        "word_checkpoint"
                    )
                    saved_w = saved_cal_prov.get("word")
                    if (
                        "S0" in saved_states
                        and isinstance(saved_states["S0"], dict)
                        and not str(
                            saved_states["S0"].get("note", "")
                        ).startswith("fallback_from_")
                        and not _is_fake_provenance(saved_w)
                        and (
                            not saved_w
                            or not cur_w
                            or str(cur_w) == "not_available"
                            or _same_checkpoint_path(saved_w, cur_w)
                        )
                    ):
                        reusable_cal_states["S0"] = dict(saved_states["S0"])

                    cur_r = ref_prov.get("revision") or resolved_ckpts.get(
                        "revision_checkpoint"
                    )
                    saved_r = saved_cal_prov.get("revision")
                    k2_lam_ok = (
                        float(
                            saved_states.get("K2", {}).get(
                                "language_weight", 0.0
                            )
                        )
                        >= min_lam_cfg - 1e-6
                    )
                    if (
                        cal_stage_ok
                        and k2_lam_ok
                        and all(
                            k in saved_states
                            and isinstance(saved_states[k], dict)
                            for k in ("K0", "K1", "K2")
                        )
                        and not _is_fake_provenance(saved_r)
                        and (
                            not saved_r
                            or not cur_r
                            or str(cur_r) == "not_available"
                            or _same_checkpoint_path(saved_r, cur_r)
                        )
                    ):
                        for k_st in ("K0", "K1", "K2"):
                            reusable_cal_states[k_st] = dict(saved_states[k_st])

                    cur_c = ref_prov.get("context") or resolved_ckpts.get(
                        "context_checkpoint"
                    )
                    saved_c = saved_cal_prov.get("context")
                    has_real_cur_c = bool(
                        cur_c and str(cur_c) != "not_available"
                    )
                    ctx_prov_ok = (
                        bool(saved_c)
                        and not _is_fake_provenance(saved_c)
                        and _same_checkpoint_path(saved_c, cur_c)
                        if has_real_cur_c
                        else not _is_fake_provenance(saved_c)
                    )
                    if (
                        "F" in saved_states
                        and isinstance(saved_states["F"], dict)
                        and not str(
                            saved_states["F"].get("note", "")
                        ).startswith("fallback_from_")
                        and ctx_prov_ok
                    ):
                        reusable_cal_states["F"] = dict(saved_states["F"])
            except Exception:
                reusable_cal_states = {}
                existing_cal_base = None

        if (
            existing_cal_base is not None
            and required_cal_states
            and all(st in reusable_cal_states for st in required_cal_states)
        ):
            calibration_res = dict(existing_cal_base)
            calibration_res["states"] = {
                st: reusable_cal_states[st] for st in required_cal_states
            }
            if ref_prov:
                calibration_res["checkpoint_provenance"] = dict(ref_prov)
            try:
                calibration_path.write_text(
                    json.dumps(calibration_res, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )
            except OSError:
                pass
        else:
            cal_source_path = (
                val_export_path
                if (args.split != "val" or val_export_path.is_file())
                else export_path
            )
            if force_rerun or not _export_matches_cms_eta(cal_source_path, cms_eta):
                _run_export_for_split("val", cal_source_path)
            val_payload = load_neural_vocabulary_scores(cal_source_path)
            val_clean = dict(val_payload.get("clean_scores") or {})
            val_prov = dict(val_payload.get("checkpoint_provenance") or {})
            if not required_cal_states:
                required_cal_states = tuple(
                    st for st in ("S0", "K0", "K1", "K2", "F") if st in val_clean
                )
            missing_cal_states = [
                st
                for st in required_cal_states
                if st not in reusable_cal_states and st in val_clean
            ]
            if any(st in missing_cal_states for st in ("K0", "K1", "K2")) or (
                shared_lam_cfg and missing_cal_states
            ):
                for k_st in ("K0", "K1", "K2"):
                    if k_st in val_clean and k_st not in missing_cal_states:
                        missing_cal_states.append(k_st)
                if shared_lam_cfg:
                    missing_cal_states = [
                        st for st in required_cal_states if st in val_clean
                    ]
            states_to_run = tuple(
                st for st in ("S0", "K0", "K1", "K2", "F") if st in missing_cal_states
            )
            if not states_to_run and existing_cal_base is not None:
                calibration_res = dict(existing_cal_base)
                calibration_res["states"] = {
                    st: reusable_cal_states[st]
                    for st in required_cal_states
                    if st in reusable_cal_states
                }
            else:
                scorer = _get_or_init_shared_scorer(
                    list(val_payload["vocabulary"])
                )
                partial_res = calibrate_state_hyperparameters(
                    val_payload,
                    scorer,
                    states=states_to_run if states_to_run else required_cal_states,
                    temperature_grid=temp_grid,
                    language_weight_grid=lambda_grid,
                    beam_width=beam_width,
                    candidate_top_k=top_k,
                    segment_batch_size=segment_batch_size,
                    metric_name=str(cal_cfg.get("metric", "composite")),
                    enforce_monotonic_lambda=bool(
                        cal_cfg.get("enforce_monotonic_lambda", True)
                    ),
                    shared_language_weight=shared_lam_cfg,
                    min_language_weight=min_lam_cfg,
                    use_calibration_partition=bool(
                        cal_cfg.get("use_calibration_partition", True)
                    ),
                    max_calibration_segments=args.max_segments
                    or cal_cfg.get("max_calibration_segments", 64),
                    output_path=None,
                    show_progress=show_progress,
                )
                merged_states: dict[str, Any] = {}
                for st in required_cal_states:
                    if st in partial_res.get("states", {}):
                        merged_states[st] = partial_res["states"][st]
                    elif st in reusable_cal_states:
                        merged_states[st] = reusable_cal_states[st]
                calibration_res = dict(partial_res)
                calibration_res["states"] = merged_states
                if existing_cal_base is not None and isinstance(
                    existing_cal_base.get("grids"), dict
                ):
                    merged_grids = dict(existing_cal_base["grids"])
                    merged_grids.update(partial_res.get("grids") or {})
                    calibration_res["grids"] = {
                        st: merged_grids[st]
                        for st in required_cal_states
                        if st in merged_grids
                    }
            calibration_res["checkpoint_provenance"] = (
                dict(ref_prov) if ref_prov else dict(val_prov)
            )
            calibration_res["revision_stage"] = str(
                val_prov.get("revision_stage") or effective_rev_stage or ""
            )
            calibration_res["cms_eta"] = float(cms_eta)
            try:
                calibration_path.parent.mkdir(parents=True, exist_ok=True)
                calibration_path.write_text(
                    json.dumps(calibration_res, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )
            except OSError:
                pass
        results_summary["calibration_path"] = str(calibration_path)
        results_summary["state_hyperparameters"] = calibration_res["states"]

    if args.step in ("decode", "all"):
        if not _export_matches_cms_eta(export_path, cms_eta):
            _run_export_for_split(args.split, export_path)
            export_was_regenerated = True
        split_payload = load_neural_vocabulary_scores(export_path)
        available_clean = split_payload.get("clean_scores", {})
        available_shifted = split_payload.get("shifted_scores", {})
        ckpt_prov = dict(split_payload.get("checkpoint_provenance") or {})
        has_real_context = "F" in available_clean and not _is_fake_provenance(
            ckpt_prov.get("context")
        )
        has_s3_clean = all(k in available_clean for k in ("K0", "K1", "K2"))
        f_mode = "F_full_context" if has_real_context else "not_available"
        if calibration_path.is_file():
            cal_json = json.loads(calibration_path.read_text(encoding="utf-8"))
            hyperparameters = cal_json
        else:
            hyperparameters = {
                "states": {
                    s: {"temperature": 0.07, "language_weight": 0.15}
                    for s in ("S0", "K0", "K1", "K2", "F")
                    if s in available_clean
                }
            }

        decoded_bundles: dict[str, Any] = {}
        multi_subject_bundles: dict[str, Any] = {}
        vocabulary_scaling_bundles: dict[str, Any] = {}
        total_split_segs = len(split_payload.get("segments") or [])
        expected_seg_count = (
            min(total_split_segs, int(args.max_segments))
            if args.max_segments is not None
            else total_split_segs
        )
        target_states_hp = dict(
            hyperparameters.get("states", hyperparameters) or {}
        )
        if not has_real_context:
            target_states_hp.pop("F", None)

        if not force_rerun and decoding_path.is_file():
            try:
                existing_dec = json.loads(
                    decoding_path.read_text(encoding="utf-8")
                )
                existing_prov = dict(
                    existing_dec.get("checkpoint_provenance") or {}
                )
                saved_hp = dict(
                    existing_dec.get("state_hyperparameters") or {}
                )

                def _hp_matches_state(st: str) -> bool:
                    if st not in target_states_hp:
                        return True
                    return (
                        isinstance(saved_hp.get(st), dict)
                        and isinstance(target_states_hp.get(st), dict)
                        and abs(
                            float(saved_hp[st].get("temperature", -1.0))
                            - float(target_states_hp[st].get("temperature", -2.0))
                        )
                        <= 1e-6
                        and abs(
                            float(saved_hp[st].get("language_weight", -1.0))
                            - float(
                                target_states_hp[st].get("language_weight", -2.0)
                            )
                        )
                        <= 1e-6
                    )

                saved_dec_stage = existing_prov.get("revision_stage")
                cur_dec_stage = ckpt_prov.get("revision_stage")
                dec_stage_ok = (
                    not saved_dec_stage
                    or not cur_dec_stage
                    or saved_dec_stage == cur_dec_stage
                )
                saved_rev_ckpt = existing_prov.get("revision")
                cur_rev_ckpt = ckpt_prov.get("revision")
                rev_ckpt_ok = not _is_fake_provenance(saved_rev_ckpt) and (
                    not saved_rev_ckpt
                    or not cur_rev_ckpt
                    or str(cur_rev_ckpt) == "not_available"
                    or _same_checkpoint_path(saved_rev_ckpt, cur_rev_ckpt)
                )
                rev_ok = bool(
                    dec_stage_ok
                    and rev_ckpt_ok
                    and all(
                        _hp_matches_state(st)
                        for st in ("K0", "K1", "K2")
                        if st in available_clean
                    )
                )

                saved_w_ckpt = existing_prov.get("word")
                cur_w_ckpt = ckpt_prov.get("word")
                word_ok = bool(
                    "S0" in available_clean
                    and not _is_fake_provenance(saved_w_ckpt)
                    and (
                        not saved_w_ckpt
                        or not cur_w_ckpt
                        or str(cur_w_ckpt) == "not_available"
                        or _same_checkpoint_path(saved_w_ckpt, cur_w_ckpt)
                    )
                    and _hp_matches_state("S0")
                )

                saved_c_ckpt = existing_prov.get("context")
                cur_c_ckpt = ckpt_prov.get("context")
                has_real_cur_c = bool(
                    cur_c_ckpt and str(cur_c_ckpt) != "not_available"
                )
                ctx_ckpt_ok = (
                    bool(saved_c_ckpt)
                    and not _is_fake_provenance(saved_c_ckpt)
                    and _same_checkpoint_path(saved_c_ckpt, cur_c_ckpt)
                    if has_real_cur_c
                    else (
                        not _is_fake_provenance(saved_c_ckpt)
                        and str(existing_dec.get("f_condition_mode", ""))
                        != "F_from_K2"
                    )
                )
                ctx_ok = bool(
                    has_real_context
                    and ctx_ckpt_ok
                    and _hp_matches_state("F")
                )

                def _cond_reusable(cond_name: str, cond_obj: Any) -> bool:
                    if not isinstance(cond_obj, dict):
                        return False
                    if int(cond_obj.get("segment_count", -1)) != expected_seg_count:
                        return False
                    if cond_name == "F_local":
                        return False
                    if cond_name == "LM-Only":
                        return True
                    if cond_name in ("S0", "S0_clean"):
                        return word_ok
                    if cond_name in ("F", "F_clean"):
                        return ctx_ok
                    if cond_name in (
                        "K0",
                        "K1",
                        "S3_clean",
                        "K0_shifted_neg75s",
                        "S3_shifted_neg75s",
                    ):
                        return rev_ok
                    return False

                if (
                    existing_dec.get("dataset") == args.dataset
                    and existing_dec.get("split") == args.split
                    and int(existing_dec.get("seed", -1)) == int(args.seed)
                ):
                    for cond_k, cond_b in (
                        existing_dec.get("decoded_bundles") or {}
                    ).items():
                        if (
                            cond_k == "S3_shifted_neg75s"
                            and isinstance(cond_b, dict)
                            and not cond_b.get("screen_histories")
                        ):
                            continue
                        if _cond_reusable(cond_k, cond_b):
                            decoded_bundles[cond_k] = cond_b
                    if (
                        rev_ok
                        and (word_ok or "S0" not in available_clean)
                        and isinstance(
                            existing_dec.get("multi_subject_bundles"), dict
                        )
                    ):
                        multi_subject_bundles = dict(
                            existing_dec["multi_subject_bundles"]
                        )
                    for scale_k, scale_b in (
                        existing_dec.get("vocabulary_scaling_bundles") or {}
                    ).items():
                        if not isinstance(scale_b, dict):
                            continue
                        conds_b = scale_b.get("conditions")
                        if not isinstance(conds_b, dict):
                            continue
                        filtered_conds = {
                            ck: cb
                            for ck, cb in conds_b.items()
                            if _cond_reusable(ck, cb)
                        }
                        if filtered_conds:
                            scale_copy = dict(scale_b)
                            scale_copy["conditions"] = filtered_conds
                            vocabulary_scaling_bundles[str(scale_k)] = scale_copy
            except Exception:
                decoded_bundles = {}
                multi_subject_bundles = {}
                vocabulary_scaling_bundles = {}

        def _ensure_scorer():
            return _get_or_init_shared_scorer(list(split_payload["vocabulary"]))

        def _persist_decoding_bundle() -> None:
            _backfill_intermediate_and_shifted_bundles(decoded_bundles)
            decoded_bundles.pop("F_local", None)
            if not has_real_context:
                decoded_bundles.pop("F", None)
                decoded_bundles.pop("F_clean", None)
            bundle_obj = {
                "schema_version": "s3-decoding-bundle/v2",
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

        any_main_decoded = False
        if "LM-Only" not in decoded_bundles:
            decoded_bundles["LM-Only"] = decode_dataset_segments(
                split_payload,
                _ensure_scorer(),
                hyperparameters,
                condition="LM-Only",
                alignment="clean",
                beam_width=beam_width,
                candidate_top_k=top_k,
                segment_batch_size=segment_batch_size,
                max_segments=args.max_segments,
                show_progress=show_progress,
                progress_desc=f"S3 Decode [LM-Only | {args.split}]",
            )
            any_main_decoded = True

        # 全部神经解码条件合并为单次同步批量束搜索：静态条件（S0/K0/K1/F/Shifted-K0）
        # 逐句段 λ 合并为一个批次，流式条件（Rolling-K2 / Shifted-K2）合并为一个批次，
        # 同一句段的不同条件相邻排入同一批次以共享语言模型前缀 KV-Cache。
        main_jobs: dict[str, dict] = {}
        for baseline_mode in ("S0", "K0", "K1"):
            if baseline_mode in available_clean and baseline_mode not in decoded_bundles:
                main_jobs[baseline_mode] = {
                    "condition": baseline_mode,
                    "alignment": "clean",
                }
        if has_s3_clean and "S3_clean" not in decoded_bundles:
            main_jobs["S3_clean"] = {"condition": "S3", "alignment": "clean"}
        if has_real_context and "F" in available_clean and "F" not in decoded_bundles:
            main_jobs["F"] = {"condition": "F", "alignment": "clean"}
        if "K0" in available_shifted and "K0_shifted_neg75s" not in decoded_bundles:
            main_jobs["K0_shifted_neg75s"] = {"condition": "K0", "alignment": "shifted"}
        if "K2" in available_shifted and "S3_shifted_neg75s" not in decoded_bundles:
            main_jobs["S3_shifted_neg75s"] = {
                "condition": "S3",
                "alignment": "shifted",
                "include_screen_history": True,
            }
        if main_jobs:
            decoded_bundles.update(
                decode_dataset_segment_jobs(
                    split_payload,
                    _ensure_scorer(),
                    hyperparameters,
                    main_jobs,
                    beam_width=beam_width,
                    candidate_top_k=top_k,
                    segment_batch_size=segment_batch_size,
                    max_segments=args.max_segments,
                    show_progress=show_progress,
                    progress_desc=(
                        f"S3 Decode [{len(main_jobs)} conditions batched | {args.split}]"
                    ),
                )
            )
            any_main_decoded = True

        if any_main_decoded:
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
            max_core_v = max(int(core_vocab_size or 250), 250)

            def _scale_needs_decode(n_val: int) -> bool:
                s_key = f"V={int(n_val)}"
                if s_key not in vocabulary_scaling_bundles:
                    return True
                conds = vocabulary_scaling_bundles[s_key].get("conditions") or {}
                req: list[str] = []
                if has_s3_clean:
                    req.append("S3_clean")
                if "S0" in available_clean:
                    req.append("S0")
                if "K0" in available_clean:
                    req.append("K0")
                if "K2" in available_shifted:
                    req.append("S3_shifted_neg75s")
                if n_val <= max_core_v or n_val == full_vocab_len:
                    if "K1" in available_clean:
                        req.append("K1")
                    if has_real_context and "F" in available_clean:
                        req.append("F")
                    if "K0" in available_shifted:
                        req.append("K0_shifted_neg75s")
                    req.append("LM-Only")
                return not all(r_k in conds for r_k in req)

            needs_truncated_decode = any(
                _scale_needs_decode(int(n)) for n in effective_scales
            )
            active_scorer = _ensure_scorer() if needs_truncated_decode else scorer

            def _on_scale_completed(
                scale_key: str, scale_bundle: dict[str, Any]
            ) -> None:
                vocabulary_scaling_bundles[scale_key] = scale_bundle

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
            if needs_truncated_decode:
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

        def _multi_subject_complete() -> bool:
            req_ms: list[str] = []
            if has_s3_clean:
                req_ms.append("S3_clean")
            if "S0" in available_clean:
                req_ms.append("S0")
            if "K0" in available_clean:
                req_ms.append("K0")
            if "K2" in available_shifted:
                req_ms.append("S3_shifted_neg75s")
            for k_key in expected_k_keys:
                if k_key not in multi_subject_bundles:
                    return False
                ms_entry = multi_subject_bundles[k_key]
                if not isinstance(ms_entry, dict):
                    return False
                ms_conds = ms_entry.get("conditions") or {}
                if not all(c_k in ms_conds for c_k in req_ms):
                    return False
                by_v = ms_entry.get("by_vocabulary_size")
                if not isinstance(by_v, dict) or not by_v:
                    return False
            return True

        if (
            len(k_observations) >= 2
            and len(unique_subjects) > 1
            and not _multi_subject_complete()
        ):
            multi_subject_bundles = run_multi_subject_observation_sweep(
                split_payload,
                _ensure_scorer(),
                hyperparameters,
                k_observations=k_observations,
                vocab_sizes=multi_subject_vocab_sizes,
                condition="S3",
                alignment="clean",
                single_subject_s3_bundle=decoded_bundles.get("S3_clean"),
                single_subject_full_vocab_bundles=decoded_bundles,
                single_subject_vocab_scaling_bundles=vocabulary_scaling_bundles,
                beam_width=beam_width,
                candidate_top_k=top_k,
                segment_batch_size=segment_batch_size,
                max_segments=args.max_segments,
                show_progress=show_progress,
            )

        _persist_decoding_bundle()
        results_summary["decoding_path"] = str(decoding_path)
        results_summary["checkpoint_provenance"] = ckpt_prov
        results_summary["f_condition_mode"] = f_mode

    if args.step in ("evaluate", "all"):
        from braindecoding.evaluation.sequence_decoding_evaluation import (
            format_sequence_decoding_report,
            summarize_and_export_multi_seed_sequence_decoding,
        )

        if args.step == "all" and "decoded_bundles" in locals():
            language = "fr" if args.dataset == "pallier2025" else "zh"
            decoded_bundles_map = dict(decoded_bundles)
            decoded_bundles_map.pop("F_local", None)
            multi_subject_bundles_map = dict(multi_subject_bundles)
            vocabulary_scaling_bundles_map = dict(vocabulary_scaling_bundles)
            if "state_hyperparameters" not in results_summary:
                results_summary["state_hyperparameters"] = target_states_hp
        else:
            bundle = json.loads(decoding_path.read_text(encoding="utf-8"))
            language = str(
                bundle.get(
                    "language", "fr" if args.dataset == "pallier2025" else "zh"
                )
            )
            bundle_prov = dict(bundle.get("checkpoint_provenance") or {})
            bundle_f_mode = str(bundle.get("f_condition_mode") or "")
            fake_ctx_in_bundle = (
                _is_fake_provenance(bundle_prov.get("context"))
                or bundle_f_mode == "F_from_K2"
            )
            if "state_hyperparameters" not in results_summary and bundle.get(
                "state_hyperparameters"
            ):
                shp = dict(bundle["state_hyperparameters"])
                if fake_ctx_in_bundle:
                    shp.pop("F", None)
                results_summary["state_hyperparameters"] = shp
            if "f_condition_mode" not in results_summary and bundle.get(
                "f_condition_mode"
            ):
                results_summary["f_condition_mode"] = (
                    "not_available"
                    if fake_ctx_in_bundle
                    else bundle["f_condition_mode"]
                )
            if "decoding_path" not in results_summary:
                results_summary["decoding_path"] = str(decoding_path)
            decoded_bundles_map = dict(bundle.get("decoded_bundles") or {})
            decoded_bundles_map.pop("F_local", None)
            if fake_ctx_in_bundle:
                decoded_bundles_map.pop("F", None)
                decoded_bundles_map.pop("F_clean", None)
            multi_subject_bundles_map = dict(
                bundle.get("multi_subject_bundles") or {}
            )
            vocabulary_scaling_bundles_map = {}
            for sk, sb in (
                bundle.get("vocabulary_scaling_bundles") or {}
            ).items():
                if isinstance(sb, dict) and fake_ctx_in_bundle:
                    sb_copy = dict(sb)
                    sb_conds = dict(sb_copy.get("conditions") or {})
                    sb_conds.pop("F", None)
                    sb_conds.pop("F_clean", None)
                    sb_copy["conditions"] = sb_conds
                    vocabulary_scaling_bundles_map[sk] = sb_copy
                else:
                    vocabulary_scaling_bundles_map[sk] = sb

        word_embedding_map = None
        if not bool(args.no_bertscore) and (
            "split_payload" in locals() or export_path.is_file()
        ):
            if "split_payload" not in locals():
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
            core_vocab_sizes=core_vocab_sizes,
            state_hyperparameters=results_summary.get("state_hyperparameters"),
            seed=int(args.seed),
            exp_tag=exp_tag,
            cms_eta=cms_eta,
            f_condition_mode=results_summary.get("f_condition_mode"),
            plot=plot_enabled,
        )
        results_summary["evaluation_dir"] = str(paths["evaluation_dir"])
        for rep_k in (
            "conditions",
            "multi_subject_scaling",
            "core_vocabulary_benchmark",
            "core_vocabulary_benchmarks",
            "full_vocab_core_subsets",
            "vocabulary_scaling",
            "c_gated_online_stability",
            "paired_statistical_significance",
            "figures",
            "tables",
            "markdown",
            "report_path",
        ):
            if rep_k in eval_report:
                results_summary[rep_k] = eval_report[rep_k]
        report_txt_path = (
            paths["evaluation_dir"] / f"s3_evaluation_report_{args.split}.txt"
        )
        report_txt_path.write_text(
            format_sequence_decoding_report(results_summary, verbose=True) + "\n",
            encoding="utf-8",
        )
        results_summary["report_txt_path"] = str(report_txt_path)

        if not bool(getattr(args, "_in_multi_seed_loop", False)):
            seq_root_dir = paths["base_dir"].parent
            multi_seed_summary = summarize_and_export_multi_seed_sequence_decoding(
                seq_root_dir,
                split=args.split,
                dataset=args.dataset,
                exp_tag=exp_tag,
                plot=plot_enabled,
            )
            results_summary["multi_seed_summary"] = multi_seed_summary

    return results_summary


def main(argv: list[str] | None = None) -> int:
    from braindecoding.evaluation.sequence_decoding_evaluation import (
        print_sequence_decoding_report,
    )

    parser = build_parser()
    args = parser.parse_args(argv)
    summary = run_sequence_decoding_pipeline(args)
    if bool(getattr(args, "json_output", False)):
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    elif isinstance(summary, dict) and isinstance(summary.get("per_seed_summaries"), dict):
        for seed_summary in summary["per_seed_summaries"].values():
            if "conditions" in seed_summary or "vocabulary_scaling" in seed_summary:
                print_sequence_decoding_report(
                    seed_summary,
                    verbose=bool(getattr(args, "verbose", False)),
                )
        ms_rep = summary.get("multi_seed_summary")
        if isinstance(ms_rep, dict) and (
            "conditions" in ms_rep or "vocabulary_scaling" in ms_rep
        ):
            print_sequence_decoding_report(
                ms_rep,
                verbose=bool(getattr(args, "verbose", False)),
            )
    elif "conditions" not in summary and "vocabulary_scaling" not in summary:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    else:
        print_sequence_decoding_report(
            summary,
            verbose=bool(getattr(args, "verbose", False)),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
