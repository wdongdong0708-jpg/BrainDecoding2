"""统一评估可视化与 Matplotlib 图表渲染模块。

集中管理词级检索评估、时间错位/全零信号负对照、多阶段对比、留一法共模扣除、
中心化 SVD 子空间相干度去偏以及 S3 序列解码的全部图表渲染，并提供从已保存 JSON
零成本重绘图表（`replot_from_saved_json`）的能力。
"""

from __future__ import annotations

from contextlib import contextmanager
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

WORKFLOW_STAGES = ("word", "context", "revision")

STAGE_DISPLAY_NAMES = {
    "word": "Stage 1: Word-Level Brain Encoder",
    "context": "Stage 2.1: Word-Level Context Transformer",
    "revision": "Stage 2.2: Local-Window Rolling Revision",
}

STAGE_SHORT_LABELS = {
    "word": "Stage 1: Word\n(CNN Only)",
    "context": "Stage 2.1: Context\n(CNN + 16L Full Window)",
    "revision": "Stage 2.2: Revision\n(CNN + 16L Local K2)",
}

EVALUATION_PALETTE = {
    "pallier2025": "#1f77b4",
    "chineseeeg2_littleprince": "#6a3d9a",
    "clean": "#1f77b4",
    "shifted": "#ff7f0e",
    "debiased": "#2ca02c",
    "highlight": "#d62728",
    "neutral": "#b0bec5",
    "chance": "#666666",
    "word": "#4e79a7",
    "context": "#f28e2b",
    "revision": "#59a14f",
    "K0": "#4e79a7",
    "K1": "#f28e2b",
    "K2": "#59a14f",
}


@contextmanager
def evaluation_plot_context():
    """提供统一、跨平台安全的 Matplotlib 评估图表渲染上下文。"""
    import matplotlib as mpl

    with mpl.rc_context(
        {
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    ):
        yield


def _save_dual_format(
    fig,
    png_path: Path,
    pdf_path: Path,
    *,
    dpi: int = 160,
    bbox_inches: str | None = None,
) -> None:
    """统一保存 PNG 与 PDF 双格式并确保关闭 Figure 释放内存。"""
    import matplotlib.pyplot as plt

    png_path.parent.mkdir(parents=True, exist_ok=True)
    save_kwargs: dict[str, Any] = {}
    if bbox_inches is not None:
        save_kwargs["bbox_inches"] = bbox_inches
    try:
        with evaluation_plot_context():
            fig.savefig(png_path, dpi=dpi, **save_kwargs)
            fig.savefig(pdf_path, **save_kwargs)
    finally:
        plt.close(fig)


def _normalize_eval_result_for_plotting(
    eval_result: Mapping[str, Any],
) -> dict[str, Any]:
    """Normalize optional/flat fields in eval_result so standalone JSON replotting never fails."""
    out = dict(eval_result)
    out.setdefault("dataset", "pallier2025")
    out.setdefault("stage", "word")
    out.setdefault("split", "test")
    out.setdefault("seed", 0)
    subset = out.get("subset")
    if not isinstance(subset, Mapping):
        out["subset"] = {
            "subset_tag": str(subset) if subset else "all",
        }
    else:
        out["subset"] = {"subset_tag": subset.get("subset_tag", subset.get("tag", "all")), **subset}
    ckpt = out.get("checkpoint")
    if not isinstance(ckpt, Mapping):
        out["checkpoint"] = {
            "path": str(ckpt or ""),
            "epoch": out.get("evaluated_epoch", out.get("checkpoint_epoch", 0)),
        }
    else:
        out["checkpoint"] = {
            "epoch": ckpt.get("epoch", out.get("evaluated_epoch", 0)),
            **ckpt,
        }
    metrics = dict(out.get("metrics") or {})
    if "vocabulary_sweep" not in metrics:
        sw: dict[str, dict[str, Any]] = {}
        if "vocab_50_macro_top10" in metrics:
            sw["N50"] = {
                "label": "50-Word (N50)",
                "vocabulary_size": 50,
                "macro_top10": float(metrics.get("vocab_50_macro_top10", 0.0)),
                "micro_top10": float(metrics.get("vocab_50_micro_top10", 0.0)),
                "macro_top5": float(metrics.get("vocab_50_macro_top5", 0.0)),
                "macro_top1": float(metrics.get("vocab_50_macro_top1", 0.0)),
                "mrr": float(metrics.get("vocab_50_mrr_macro", 0.0)),
                "chance_top10": 0.2,
            }
        if "full_vocab_macro_top10" in metrics:
            fv_sz = int(metrics.get("full_vocab_size", 2020))
            sw["full_vocab"] = {
                "label": f"Full Vocab ({fv_sz})",
                "vocabulary_size": fv_sz,
                "macro_top10": float(metrics.get("full_vocab_macro_top10", 0.0)),
                "micro_top10": float(metrics.get("full_vocab_micro_top10", 0.0)),
                "macro_top5": float(metrics.get("full_vocab_macro_top5", 0.0)),
                "macro_top1": float(metrics.get("full_vocab_macro_top1", 0.0)),
                "mrr": float(metrics.get("full_vocab_mrr_macro", 0.0)),
                "chance_top10": min(10.0, float(fv_sz)) / max(1.0, float(fv_sz)),
            }
        metrics["vocabulary_sweep"] = sw
    if "per_subject" not in metrics and isinstance(metrics.get("per_subject_vocab50"), Mapping):
        metrics["per_subject"] = [
            {
                "subject_id": str(sid),
                "macro_top10": float(s.get("macro_top10", 0.0)),
                "macro_top1": float(s.get("macro_top1", 0.0)),
            }
            for sid, s in metrics["per_subject_vocab50"].items()
            if isinstance(s, Mapping)
        ]
    if isinstance(metrics.get("revision_horizons"), Mapping):
        norm_hz: dict[str, dict[str, Any]] = {}
        for hk, hv in metrics["revision_horizons"].items():
            if isinstance(hv, Mapping):
                h_copy = dict(hv)
                h_copy.setdefault("label", f"{hk} (+{h_copy.get('delay_words', 0)}w)")
                h_copy.setdefault("common_full_vocab_macro_top10", 0.0)
                norm_hz[str(hk)] = h_copy
        metrics["revision_horizons"] = norm_hz
    out["metrics"] = metrics

    neg = out.get("negative_controls") or out.get("negative_control")
    if isinstance(neg, Mapping):
        neg_copy = dict(neg)
        clean_ref = dict(neg_copy.get("clean_reference") or {})
        clean_ref.setdefault("label", "Clean (0s)")
        clean_ref.setdefault("vocab50_chance_top10", 0.2)
        neg_copy["clean_reference"] = clean_ref
        conds = neg_copy.get("conditions")
        if isinstance(conds, Mapping):
            cond_list = []
            for ck, cv in conds.items():
                if isinstance(cv, Mapping):
                    s_row = dict(cv.get("summary") or {})
                    s_row.setdefault("key", str(ck))
                    s_row.setdefault("label", str(ck))
                    if "shift_seconds" in cv and "shift_seconds" not in s_row:
                        s_row["shift_seconds"] = cv["shift_seconds"]
                    if "delta_vs_clean" not in s_row:
                        s_row["delta_vs_clean"] = {
                            "vocab50_macro_top10": float(s_row.get("vocab50_macro_top10", 0.0))
                            - float(clean_ref.get("vocab50_macro_top10", 0.0)),
                            "full_vocab_macro_top10": float(s_row.get("full_vocab_macro_top10", 0.0))
                            - float(clean_ref.get("full_vocab_macro_top10", 0.0)),
                        }
                    cond_list.append({"key": str(ck), **cv, "summary": s_row})
            neg_copy["conditions"] = cond_list
        out["negative_controls"] = neg_copy
    return out


def plot_negative_control_figures(
    eval_result: Mapping[str, Any],
    figures_dir: str | Path,
) -> dict[str, str]:
    """绘制单阶段时间错位（-15s..-120s）/ 全零信号负对照对比图（PNG 与 PDF）。"""
    eval_result = _normalize_eval_result_for_plotting(eval_result)
    neg_ctrl = eval_result.get("negative_controls") or {}
    ctrl_conditions = neg_ctrl.get("conditions") or []
    if not ctrl_conditions:
        return {}

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = Path(figures_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset = str(eval_result["dataset"])
    stage = str(eval_result["stage"])
    stage_title = STAGE_DISPLAY_NAMES.get(stage, stage)
    split = str(eval_result["split"]).upper()
    subset_tag = eval_result["subset"]["subset_tag"]
    seed = eval_result["seed"]
    clean_row = neg_ctrl["clean_reference"]

    neg_shifts = sorted(
        [
            c["summary"]
            for c in ctrl_conditions
            if c["summary"].get("shift_seconds") is not None
            and float(c["summary"]["shift_seconds"]) < 0
        ],
        key=lambda r: float(r["shift_seconds"]),
    )
    pos_shifts = sorted(
        [
            c["summary"]
            for c in ctrl_conditions
            if c["summary"].get("shift_seconds") is not None
            and float(c["summary"]["shift_seconds"]) > 0
        ],
        key=lambda r: float(r["shift_seconds"]),
    )
    other_ctrls = [
        c["summary"]
        for c in ctrl_conditions
        if c["summary"].get("shift_seconds") is None
    ]
    ordered_rows = [*neg_shifts, clean_row, *pos_shifts, *other_ctrls]
    clean_idx = len(neg_shifts)

    labels = [r["label"].replace("Shift ", "") for r in ordered_rows]
    x = np.arange(len(ordered_rows))
    v50_macro10 = [r["vocab50_macro_top10"] * 100.0 for r in ordered_rows]
    v50_micro10 = [r["vocab50_micro_top10"] * 100.0 for r in ordered_rows]
    v50_macro1 = [r["vocab50_macro_top1"] * 100.0 for r in ordered_rows]
    chance_50 = float(clean_row.get("vocab50_chance_top10", 0.2)) * 100.0

    primary_color = EVALUATION_PALETTE.get(dataset, "#1f77b4")
    ctrl_color = EVALUATION_PALETTE["neutral"]
    highlight_color = EVALUATION_PALETTE["highlight"]

    with evaluation_plot_context():
        fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(15.0, 5.8), dpi=160)
        fig.suptitle(
            f"Negative Control Evaluation ({split})  |  {dataset}  |  {stage_title}  |  subset={subset_tag} (seed={seed})",
            fontsize=12.5,
            fontweight="bold",
            y=0.98,
        )

        bar_colors = [
            primary_color if idx == clean_idx else ctrl_color
            for idx in range(len(ordered_rows))
        ]
        bars = ax_left.bar(
            x,
            v50_macro10,
            width=0.56,
            color=bar_colors,
            edgecolor="#263238",
            linewidth=0.9,
            alpha=0.92,
            label="50-Word Macro Top-10 (%)",
        )
        ax_left.plot(
            x,
            v50_micro10,
            "-s",
            color=EVALUATION_PALETTE["shifted"],
            linewidth=1.8,
            markersize=5.5,
            label="50-Word Micro Top-10 (%)",
        )
        ax_left.plot(
            x,
            v50_macro1,
            "--^",
            color=EVALUATION_PALETTE["debiased"],
            linewidth=1.6,
            markersize=5.0,
            label="50-Word Macro Top-1 (%)",
        )
        ax_left.axhline(
            chance_50,
            color="black",
            linestyle=":",
            linewidth=1.4,
            label=f"Random Chance Top-10 ({chance_50:.1f}%)",
        )
        for idx, bar in enumerate(bars):
            h = bar.get_height()
            ax_left.annotate(
                f"{h:.1f}%",
                xy=(bar.get_x() + bar.get_width() / 2, h),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8.5,
                fontweight="bold" if idx == clean_idx else "normal",
                color=primary_color if idx == clean_idx else "#37474f",
            )
        ax_left.set_title(
            f"(A) 50-Word Retrieval Accuracy: Clean (0s) vs Shifted Signals ({split})",
            fontsize=11,
            fontweight="bold",
        )
        ax_left.set_xticks(x)
        ax_left.set_xticklabels(labels, fontsize=9.5)
        ax_left.set_ylabel("Accuracy (%)", fontsize=10)
        ax_left.set_ylim(
            0, max(max(v50_macro10 + v50_micro10, default=35.0) * 1.24, 30.0)
        )
        ax_left.grid(axis="y", linestyle=":", alpha=0.5)
        ax_left.legend(loc="upper right", fontsize=8.5)

        has_horizons = all("horizons" in r for r in ordered_rows)
        if has_horizons and stage == "revision":
            for h_key in ("K0", "K1", "K2"):
                h_vals = [
                    r["horizons"]
                    .get(h_key, {})
                    .get("common_vocab50_macro_top10", 0.0)
                    * 100.0
                    for r in ordered_rows
                ]
                ax_right.plot(
                    x,
                    h_vals,
                    "-o",
                    linewidth=2.1,
                    markersize=6,
                    color=EVALUATION_PALETTE[h_key],
                    label=f"{h_key} Common 50-Word Macro Top-10 (%)",
                )
            ax_right.axhline(
                chance_50,
                color="black",
                linestyle=":",
                linewidth=1.3,
                label=f"Chance Top-10 ({chance_50:.1f}%)",
            )
            ax_right.axvline(
                clean_idx,
                color=highlight_color,
                linestyle="--",
                linewidth=1.1,
                alpha=0.6,
            )
            ax_right.set_title(
                f"(B) Rolling Revision Horizons (K0 / K1 / K2) Across Signal Shifts ({split})",
                fontsize=11,
                fontweight="bold",
            )
            ax_right.set_xticks(x)
            ax_right.set_xticklabels(labels, fontsize=9.5)
            ax_right.set_ylabel("50-Word Macro Top-10 (%)", fontsize=10)
            ax_right.set_ylim(0, max(ax_right.get_ylim()[1] * 1.18, 30.0))
            ax_right.grid(True, linestyle=":", alpha=0.5)
            ax_right.legend(loc="upper right", fontsize=8.5)
        else:
            ctrl_only = [c["summary"] for c in ctrl_conditions]
            cx = np.arange(len(ctrl_only))
            c_labels = [r["label"].replace("Shift ", "") for r in ctrl_only]
            drops_50 = [
                r["delta_vs_clean"]["vocab50_macro_top10"] * 100.0
                for r in ctrl_only
            ]
            drops_fv = [
                r["delta_vs_clean"]["full_vocab_macro_top10"] * 100.0
                for r in ctrl_only
            ]
            bw = 0.36
            b1 = ax_right.bar(
                cx - bw / 2,
                drops_50,
                bw,
                color="#e15759",
                alpha=0.9,
                label="Delta 50-Word Macro Top-10 (Control - Clean)",
            )
            b2 = ax_right.bar(
                cx + bw / 2,
                drops_fv,
                bw,
                color="#76b7b2",
                alpha=0.9,
                label="Delta Full-Vocab Macro Top-10 (Control - Clean)",
            )
            ax_right.axhline(0.0, color="black", linewidth=1.2)
            for bar in list(b1) + list(b2):
                h = bar.get_height()
                va = "top" if h < 0 else "bottom"
                y_off = -11 if h < 0 else 3
                ax_right.annotate(
                    f"{h:+.1f}%",
                    xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, y_off),
                    textcoords="offset points",
                    ha="center",
                    va=va,
                    fontsize=8,
                    fontweight="bold",
                )
            ax_right.set_title(
                f"(B) Retrieval Accuracy Drop Under Negative Controls ({split})",
                fontsize=11,
                fontweight="bold",
            )
            ax_right.set_xticks(cx)
            ax_right.set_xticklabels(c_labels, fontsize=9.5)
            ax_right.set_ylabel("Accuracy Change vs Clean (%)", fontsize=10)
            y_min = min(min(drops_50 + drops_fv, default=-10.0) * 1.28, -5.0)
            y_max = max(max(drops_50 + drops_fv, default=2.0) * 1.25, 5.0)
            ax_right.set_ylim(y_min, y_max)
            ax_right.grid(axis="y", linestyle=":", alpha=0.5)
            ax_right.legend(loc="lower right", fontsize=8.5)

        fig.tight_layout(rect=[0, 0, 1, 0.93])

    split_lower = str(eval_result["split"])
    png_path = out_dir / f"negative_control_{split_lower}.png"
    pdf_path = out_dir / f"negative_control_{split_lower}.pdf"
    _save_dual_format(fig, png_path, pdf_path, dpi=160)
    return {
        "negative_control_png": str(png_path),
        "negative_control_pdf": str(pdf_path),
    }


def plot_stage_control_comparison_figure(
    stage_results: Mapping[str, Mapping[str, Any]],
    figures_dir: str | Path,
    *,
    split: str = "val",
) -> dict[str, str]:
    """绘制多阶段（Stage 1 / 2.1 / 2.2）在 Clean (0s) 与各时间错位负对照下的对比图。"""
    stage_results = {
        k: _normalize_eval_result_for_plotting(v)
        for k, v in stage_results.items()
        if isinstance(v, Mapping)
    }
    ordered_stages = [
        s
        for s in WORKFLOW_STAGES
        if s in stage_results
        and (stage_results[s].get("negative_controls") or {}).get("conditions")
    ]
    if not ordered_stages:
        return {}

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = Path(figures_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ref_res = stage_results[ordered_stages[-1]]
    dataset = ref_res["dataset"]
    subset_tag = ref_res["subset"]["subset_tag"]
    seed = ref_res["seed"]
    split_upper = split.upper()

    first_nc = ref_res["negative_controls"]
    first_conds = first_nc["conditions"]
    ref_cond_keys = {c["key"] for c in first_conds}
    ordered_stages = [
        s
        for s in ordered_stages
        if {
            c["key"]
            for c in (stage_results[s].get("negative_controls") or {}).get(
                "conditions", ()
            )
        }
        & ref_cond_keys
    ]
    if not ordered_stages:
        return {}

    neg_shifts = sorted(
        [
            c["summary"]
            for c in first_conds
            if c["summary"].get("shift_seconds") is not None
            and float(c["summary"]["shift_seconds"]) < 0
        ],
        key=lambda r: float(r["shift_seconds"]),
    )
    pos_shifts = sorted(
        [
            c["summary"]
            for c in first_conds
            if c["summary"].get("shift_seconds") is not None
            and float(c["summary"]["shift_seconds"]) > 0
        ],
        key=lambda r: float(r["shift_seconds"]),
    )
    other_ctrls = [
        c["summary"]
        for c in first_conds
        if c["summary"].get("shift_seconds") is None
    ]
    ordered_keys = [
        *(r["key"] for r in neg_shifts),
        "clean_0s",
        *(r["key"] for r in pos_shifts),
        *(r["key"] for r in other_ctrls),
    ]
    x_labels = [
        *(r["label"].replace("Shift ", "") for r in neg_shifts),
        "Clean (0s)",
        *(r["label"].replace("Shift ", "") for r in pos_shifts),
        *(r["label"] for r in other_ctrls),
    ]
    x = np.arange(len(ordered_keys))

    with evaluation_plot_context():
        fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(15.0, 5.8), dpi=160)
        fig.suptitle(
            f"Multi-Stage Negative Control Comparison ({split_upper})  |  dataset={dataset}  |  subset={subset_tag} (seed={seed})",
            fontsize=13,
            fontweight="bold",
            y=0.98,
        )

        for st in ordered_stages:
            nc = stage_results[st]["negative_controls"]
            row_map = {"clean_0s": nc["clean_reference"]}
            for c in nc["conditions"]:
                row_map[c["key"]] = c["summary"]
            y_50 = [
                float(row_map.get(k, {}).get("vocab50_macro_top10", 0.0)) * 100.0
                for k in ordered_keys
            ]
            y_fv = [
                float(row_map.get(k, {}).get("full_vocab_macro_top10", 0.0)) * 100.0
                for k in ordered_keys
            ]
            col = EVALUATION_PALETTE.get(st, "#333333")
            ax_left.plot(
                x,
                y_50,
                "-o",
                linewidth=2.2,
                markersize=6,
                color=col,
                label=STAGE_DISPLAY_NAMES.get(st, st),
            )
            ax_right.plot(
                x,
                y_fv,
                "-s",
                linewidth=2.2,
                markersize=6,
                color=col,
                label=STAGE_DISPLAY_NAMES.get(st, st),
            )

        chance_50 = (
            float(first_nc["clean_reference"].get("vocab50_chance_top10", 0.2))
            * 100.0
        )
        ax_left.axhline(
            chance_50,
            color="black",
            linestyle=":",
            linewidth=1.3,
            label=f"Random Chance ({chance_50:.1f}%)",
        )
        ax_left.set_title(
            f"(A) 50-Word Macro Top-10 Across Temporal Shifts ({split_upper})",
            fontsize=11,
            fontweight="bold",
        )
        ax_left.set_xticks(x)
        ax_left.set_xticklabels(x_labels, fontsize=9.5)
        ax_left.set_ylabel("50-Word Macro Top-10 (%)", fontsize=10)
        ax_left.set_ylim(0, max(ax_left.get_ylim()[1] * 1.18, 35.0))
        ax_left.grid(True, linestyle=":", alpha=0.5)
        ax_left.legend(loc="upper right", fontsize=8.5)

        ax_right.set_title(
            f"(B) Full-Vocabulary Macro Top-10 Across Temporal Shifts ({split_upper})",
            fontsize=11,
            fontweight="bold",
        )
        ax_right.set_xticks(x)
        ax_right.set_xticklabels(x_labels, fontsize=9.5)
        ax_right.set_ylabel("Full-Vocabulary Macro Top-10 (%)", fontsize=10)
        ax_right.set_ylim(0, max(ax_right.get_ylim()[1] * 1.22, 8.0))
        ax_right.grid(True, linestyle=":", alpha=0.5)
        ax_right.legend(loc="upper right", fontsize=8.5)

        fig.tight_layout(rect=[0, 0, 1, 0.93])

    png_path = out_dir / f"stage_control_comparison_{split}.png"
    pdf_path = out_dir / f"stage_control_comparison_{split}.pdf"
    _save_dual_format(fig, png_path, pdf_path, dpi=160)
    return {
        "stage_control_comparison_png": str(png_path),
        "stage_control_comparison_pdf": str(pdf_path),
    }


def plot_evaluation_figures(
    eval_result: Mapping[str, Any],
    figures_dir: str | Path,
) -> dict[str, str]:
    """生成单阶段 2x2 综合评估看板图（PNG 与 PDF），返回生成的文件路径字典。"""
    eval_result = _normalize_eval_result_for_plotting(eval_result)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = Path(figures_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset = str(eval_result["dataset"])
    stage = str(eval_result["stage"])
    stage_title = STAGE_DISPLAY_NAMES.get(stage, stage)
    split = str(eval_result["split"]).upper()
    subset_tag = eval_result["subset"]["subset_tag"]
    seed = eval_result["seed"]
    metrics = eval_result["metrics"]
    sweep = metrics["vocabulary_sweep"]
    per_subject = metrics.get("per_subject") or []
    horizons = metrics.get("revision_horizons") or {}
    history = eval_result.get("training_history") or []

    primary_color = EVALUATION_PALETTE.get(dataset, "#1f77b4")
    secondary_color = EVALUATION_PALETTE["shifted"]
    accent_color = EVALUATION_PALETTE["debiased"]

    with evaluation_plot_context():
        fig, axes = plt.subplots(2, 2, figsize=(15, 10.5), dpi=160)
        fig.suptitle(
            f"BrainDecoding {split} Evaluation Dashboard  |  {dataset}  |  {stage_title}\n"
            f"Subset: {subset_tag} (seed={seed}, checkpoint epoch={eval_result['checkpoint']['epoch']})",
            fontsize=13,
            fontweight="bold",
            y=0.98,
        )

        # Panel A: Retrieval Accuracy Across Vocabulary Sizes
        ax_a = axes[0, 0]
        keys = [
            k for k in ("N20", "N50", "N100", "N150", "full_vocab") if k in sweep
        ]
        x = np.arange(len(keys))
        width = 0.26
        macro_top10 = [sweep[k]["macro_top10"] * 100.0 for k in keys]
        micro_top10 = [sweep[k]["micro_top10"] * 100.0 for k in keys]
        macro_top1 = [sweep[k]["macro_top1"] * 100.0 for k in keys]
        chance_top10 = [sweep[k]["chance_top10"] * 100.0 for k in keys]
        labels = [sweep[k]["label"].replace(" (", "\n(") for k in keys]

        bars1 = ax_a.bar(
            x - width,
            macro_top10,
            width,
            label="Macro Top-10 (%)",
            color=primary_color,
            alpha=0.9,
        )
        ax_a.bar(
            x,
            micro_top10,
            width,
            label="Micro Top-10 (%)",
            color=secondary_color,
            alpha=0.85,
        )
        ax_a.bar(
            x + width,
            macro_top1,
            width,
            label="Macro Top-1 (%)",
            color=accent_color,
            alpha=0.85,
        )
        ax_a.plot(
            x,
            chance_top10,
            "k--o",
            linewidth=1.4,
            markersize=4,
            label="Random Chance (Top-10)",
        )

        for bar in bars1:
            h = bar.get_height()
            ax_a.annotate(
                f"{h:.1f}%",
                xy=(bar.get_x() + bar.get_width() / 2, h),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
                fontweight="bold",
            )
        ax_a.set_title(
            f"(A) Retrieval Accuracy by Vocabulary Size ({split})",
            fontsize=11,
            fontweight="bold",
        )
        ax_a.set_xticks(x)
        ax_a.set_xticklabels(labels, fontsize=9)
        ax_a.set_ylabel("Accuracy (%)", fontsize=10)
        ax_a.set_ylim(
            0,
            max(
                max(macro_top10 + micro_top10 + chance_top10, default=50.0) * 1.18,
                20.0,
            ),
        )
        ax_a.grid(axis="y", linestyle=":", alpha=0.5)
        ax_a.legend(loc="upper right", fontsize=8.5)

        # Panel B: Per-Subject 50-Word Decoding Accuracy
        ax_b = axes[0, 1]
        if per_subject:
            subj_ids = [s["subject_id"] for s in per_subject]
            s_top10 = [s["macro_top10"] * 100.0 for s in per_subject]
            s_top1 = [s["macro_top1"] * 100.0 for s in per_subject]
            sx = np.arange(len(subj_ids))
            sw = 0.36
            b_s10 = ax_b.bar(
                sx - sw / 2,
                s_top10,
                sw,
                label="50-Word Macro Top-10 (%)",
                color=primary_color,
                alpha=0.88,
            )
            ax_b.bar(
                sx + sw / 2,
                s_top1,
                sw,
                label="50-Word Macro Top-1 (%)",
                color=accent_color,
                alpha=0.85,
            )
            mean_top10 = float(np.mean(s_top10))
            ax_b.axhline(
                mean_top10,
                color=primary_color,
                linestyle="-.",
                linewidth=1.3,
                label=f"Mean Top-10 ({mean_top10:.1f}%)",
            )
            ax_b.axhline(
                20.0,
                color="gray",
                linestyle="--",
                linewidth=1.2,
                label="Chance Top-10 (20.0%)",
            )
            for bar in b_s10:
                h = bar.get_height()
                ax_b.annotate(
                    f"{h:.1f}",
                    xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 2),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=7.5,
                )
            ax_b.set_xticks(sx)
            ax_b.set_xticklabels(
                subj_ids, rotation=25 if len(subj_ids) > 6 else 0, fontsize=9
            )
            ax_b.set_ylim(0, max(max(s_top10, default=30.0) * 1.2, 35.0))
            ax_b.legend(loc="upper right", fontsize=8.5)
        else:
            ax_b.text(
                0.5,
                0.5,
                "No per-subject data available",
                ha="center",
                va="center",
                transform=ax_b.transAxes,
            )
        ax_b.set_title(
            f"(B) Per-Subject 50-Word Retrieval Accuracy ({split})",
            fontsize=11,
            fontweight="bold",
        )
        ax_b.set_ylabel("Accuracy (%)", fontsize=10)
        ax_b.grid(axis="y", linestyle=":", alpha=0.5)

        # Panel C: Training Convergence Curve
        ax_c = axes[1, 0]
        if history:
            ep_nums = [int(r["epoch"]) for r in history]
            tr_loss = [float(r["train_loss"]) for r in history]
            val_50 = [
                float(r.get("selection_score_legacy50", 0.0)) * 100.0
                for r in history
            ]
            val_full = [
                float(r.get("selection_score_full_vocab", 0.0)) * 100.0
                for r in history
            ]

            ax_c_right = ax_c.twinx()
            l1 = ax_c.plot(
                ep_nums,
                tr_loss,
                "-o",
                color=EVALUATION_PALETTE["highlight"],
                linewidth=1.8,
                markersize=4,
                label="Train Loss",
            )
            l2 = ax_c_right.plot(
                ep_nums,
                val_50,
                "-s",
                color=primary_color,
                linewidth=2.0,
                markersize=4.5,
                label="Val 50-Word Top-10 (%)",
            )
            l3 = ax_c_right.plot(
                ep_nums,
                val_full,
                "--^",
                color=accent_color,
                linewidth=1.6,
                markersize=4,
                label="Val Full-Vocab Top-10 (%)",
            )

            best_ep = int(eval_result["checkpoint"].get("epoch", 0))
            if best_ep in ep_nums:
                ax_c.axvline(
                    best_ep,
                    color="black",
                    linestyle=":",
                    linewidth=1.3,
                    label=f"Best Epoch ({best_ep})",
                )

            ax_c.set_xlabel("Epoch", fontsize=10)
            ax_c.set_ylabel(
                "Train Loss (SigLIP)",
                color=EVALUATION_PALETTE["highlight"],
                fontsize=10,
            )
            ax_c_right.set_ylabel(
                "Validation Top-10 Accuracy (%)", color=primary_color, fontsize=10
            )
            lines = l1 + l2 + l3
            labels_c = [line.get_label() for line in lines]
            ax_c.legend(lines, labels_c, loc="center right", fontsize=8.5)
            ax_c.grid(True, linestyle=":", alpha=0.5)
        else:
            ax_c.text(
                0.5,
                0.5,
                "Training history not found in output directory\n(Checkpoint evaluated standalone)",
                ha="center",
                va="center",
                fontsize=10,
                transform=ax_c.transAxes,
            )
        ax_c.set_title(
            "(C) Training Loss & Validation Accuracy Progression",
            fontsize=11,
            fontweight="bold",
        )

        # Panel D: Rolling Revision Horizons OR Detailed Metric Summary
        ax_d = axes[1, 1]
        if horizons and all(k in horizons for k in ("K0", "K1", "K2")):
            h_keys = ["K0", "K1", "K2"]
            hx = np.arange(len(h_keys))
            hw = 0.34
            h_50 = [
                horizons[k]["common_vocab50_macro_top10"] * 100.0 for k in h_keys
            ]
            h_full = [
                horizons[k]["common_full_vocab_macro_top10"] * 100.0
                for k in h_keys
            ]
            h_labels = [horizons[k]["label"] for k in h_keys]

            b_h1 = ax_d.bar(
                hx - hw / 2,
                h_50,
                hw,
                label="50-Word Common Macro Top-10 (%)",
                color=primary_color,
                alpha=0.9,
            )
            b_h2 = ax_d.bar(
                hx + hw / 2,
                h_full,
                hw,
                label="Full-Vocab Common Top-10 (%)",
                color=secondary_color,
                alpha=0.88,
            )
            for bar in list(b_h1) + list(b_h2):
                h = bar.get_height()
                ax_d.annotate(
                    f"{h:.2f}%",
                    xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=8.5,
                    fontweight="bold",
                )
            delta_50 = h_50[-1] - h_50[0]
            ax_d.set_title(
                f"(D) Stage 2.2 Rolling Revision Gain: K2 - K0 = {delta_50:+.2f}% (50-Word)",
                fontsize=11,
                fontweight="bold",
            )
            ax_d.set_xticks(hx)
            ax_d.set_xticklabels(h_labels, fontsize=9.5)
            ax_d.set_ylabel("Macro Top-10 Accuracy (%)", fontsize=10)
            ax_d.set_ylim(0, max(max(h_50 + h_full, default=30.0) * 1.22, 25.0))
            ax_d.legend(loc="upper left", fontsize=8.5)
            ax_d.grid(axis="y", linestyle=":", alpha=0.5)
        else:
            n50 = sweep.get("N50", {})
            fv = sweep.get("full_vocab", {})
            metric_names = [
                "Macro Top-10",
                "Micro Top-10",
                "Macro Top-5",
                "Macro Top-1",
                "MRR",
            ]
            vals_50 = [
                n50.get("macro_top10", 0.0) * 100.0,
                n50.get("micro_top10", 0.0) * 100.0,
                n50.get("macro_top5", 0.0) * 100.0,
                n50.get("macro_top1", 0.0) * 100.0,
                n50.get("mrr", 0.0) * 100.0,
            ]
            vals_fv = [
                fv.get("macro_top10", 0.0) * 100.0,
                fv.get("micro_top10", 0.0) * 100.0,
                fv.get("macro_top5", 0.0) * 100.0,
                fv.get("macro_top1", 0.0) * 100.0,
                fv.get("mrr", 0.0) * 100.0,
            ]
            mx = np.arange(len(metric_names))
            mw = 0.35
            bm1 = ax_d.bar(
                mx - mw / 2,
                vals_50,
                mw,
                label="50-Word Candidate Pool",
                color=primary_color,
                alpha=0.9,
            )
            bm2 = ax_d.bar(
                mx + mw / 2,
                vals_fv,
                mw,
                label="Full Training Vocabulary",
                color=secondary_color,
                alpha=0.88,
            )
            for bar in list(bm1) + list(bm2):
                h = bar.get_height()
                ax_d.annotate(
                    f"{h:.1f}%",
                    xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 2),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )
            ax_d.set_title(
                f"(D) Detailed Retrieval Profile: 50-Word vs Full Vocabulary ({split})",
                fontsize=11,
                fontweight="bold",
            )
            ax_d.set_xticks(mx)
            ax_d.set_xticklabels(metric_names, fontsize=9)
            ax_d.set_ylabel("Score (%)", fontsize=10)
            ax_d.set_ylim(0, max(max(vals_50 + vals_fv, default=30.0) * 1.22, 25.0))
            ax_d.legend(loc="upper right", fontsize=8.5)
            ax_d.grid(axis="y", linestyle=":", alpha=0.5)

        fig.tight_layout(rect=[0, 0, 1, 0.94])

    png_path = out_dir / f"evaluation_{eval_result['split']}_overview.png"
    pdf_path = out_dir / f"evaluation_{eval_result['split']}_overview.pdf"
    _save_dual_format(fig, png_path, pdf_path, dpi=160)
    result_paths = {
        "overview_png": str(png_path),
        "overview_pdf": str(pdf_path),
    }
    result_paths.update(plot_negative_control_figures(eval_result, out_dir))
    return result_paths


def plot_stage_comparison_figure(
    stage_results: Mapping[str, Mapping[str, Any]],
    figures_dir: str | Path,
    *,
    split: str = "val",
) -> dict[str, str]:
    """生成阶段 1 (word)、阶段 2.1 (context) 与阶段 2.2 (revision) 的横向对比图。"""
    stage_results = {
        k: _normalize_eval_result_for_plotting(v)
        for k, v in stage_results.items()
        if isinstance(v, Mapping)
    }
    ordered_stages = [s for s in WORKFLOW_STAGES if s in stage_results]
    if not ordered_stages:
        return {}

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = Path(figures_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    first_res = stage_results[ordered_stages[0]]
    dataset = first_res["dataset"]
    subset_tag = first_res["subset"]["subset_tag"]
    seed = first_res["seed"]
    split_upper = split.upper()

    with evaluation_plot_context():
        fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(14.5, 5.8), dpi=160)
        fig.suptitle(
            f"BrainDecoding Multi-Stage Comparison ({split_upper})  |  dataset={dataset}  |  subset={subset_tag} (seed={seed})",
            fontsize=13,
            fontweight="bold",
            y=0.98,
        )

        metric_defs = [
            ("N50", "macro_top10", "50-Word\nMacro Top-10"),
            ("N50", "micro_top10", "50-Word\nMicro Top-10"),
            ("N50", "macro_top1", "50-Word\nMacro Top-1"),
            ("full_vocab", "macro_top10", "Full-Vocab\nMacro Top-10"),
        ]
        mx = np.arange(len(metric_defs))
        bar_w = 0.75 / max(1, len(ordered_stages))

        for idx, st in enumerate(ordered_stages):
            res = stage_results[st]
            sw = res["metrics"]["vocabulary_sweep"]
            vals = [
                float(sw.get(v_key, {}).get(m_key, 0.0)) * 100.0
                for v_key, m_key, _ in metric_defs
            ]
            offset = (idx - (len(ordered_stages) - 1) / 2.0) * bar_w
            bars = ax_left.bar(
                mx + offset,
                vals,
                bar_w,
                label=STAGE_DISPLAY_NAMES.get(st, st),
                color=EVALUATION_PALETTE.get(st, "#333333"),
                alpha=0.9,
            )
            for bar in bars:
                h = bar.get_height()
                ax_left.annotate(
                    f"{h:.1f}%",
                    xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    fontweight="bold",
                )

        ax_left.set_title(
            f"(A) Stage Comparison on Core Retrieval Metrics ({split_upper})",
            fontsize=11,
            fontweight="bold",
        )
        ax_left.set_xticks(mx)
        ax_left.set_xticklabels([label for _, _, label in metric_defs], fontsize=9.5)
        ax_left.set_ylabel("Accuracy (%)", fontsize=10)
        ax_left.set_ylim(0, max(40.0, ax_left.get_ylim()[1] * 1.18))
        ax_left.grid(axis="y", linestyle=":", alpha=0.5)
        ax_left.legend(loc="upper right", fontsize=8.5)

        scale_keys = ["N20", "N50", "N100", "N150", "full_vocab"]
        sx = np.arange(len(scale_keys))
        scale_labels = ["N=20", "N=50", "N=100", "N=150", "Full Vocab"]

        for st in ordered_stages:
            res = stage_results[st]
            sw = res["metrics"]["vocabulary_sweep"]
            y_vals = [
                float(sw.get(k, {}).get("macro_top10", 0.0)) * 100.0
                for k in scale_keys
            ]
            ax_right.plot(
                sx,
                y_vals,
                "-o",
                linewidth=2.2,
                markersize=6,
                color=EVALUATION_PALETTE.get(st, "#333333"),
                label=STAGE_SHORT_LABELS.get(st, st).replace("\n", " "),
            )
            for x_i, y_v in zip(sx, y_vals):
                ax_right.annotate(
                    f"{y_v:.1f}%",
                    xy=(x_i, y_v),
                    xytext=(0, 6),
                    textcoords="offset points",
                    ha="center",
                    fontsize=8,
                )

        first_sw = first_res["metrics"]["vocabulary_sweep"]
        chance_curve = [
            float(first_sw.get(k, {}).get("chance_top10", 0.0)) * 100.0
            for k in scale_keys
        ]
        ax_right.plot(
            sx, chance_curve, "k--", linewidth=1.3, label="Random Chance (Top-10)"
        )

        ax_right.set_title(
            f"(B) Macro Top-10 Scaling Across Vocabulary Sizes ({split_upper})",
            fontsize=11,
            fontweight="bold",
        )
        ax_right.set_xticks(sx)
        ax_right.set_xticklabels(scale_labels, fontsize=9.5)
        ax_right.set_ylabel("Macro Top-10 Accuracy (%)", fontsize=10)
        ax_right.set_ylim(0, max(60.0, ax_right.get_ylim()[1] * 1.15))
        ax_right.grid(True, linestyle=":", alpha=0.5)
        ax_right.legend(loc="upper right", fontsize=8.5)

        fig.tight_layout(rect=[0, 0, 1, 0.93])

    png_path = out_dir / f"stage_comparison_{split}.png"
    pdf_path = out_dir / f"stage_comparison_{split}.pdf"
    _save_dual_format(fig, png_path, pdf_path, dpi=160)
    result_paths = {
        "stage_comparison_png": str(png_path),
        "stage_comparison_pdf": str(pdf_path),
    }
    result_paths.update(
        plot_stage_control_comparison_figure(stage_results, out_dir, split=split)
    )
    return result_paths


def plot_common_mode_subtraction_figures(
    report: Mapping[str, Any],
    figures_dir: str | Path,
) -> dict[str, str]:
    """生成留一法共模扣除（eta 扫描）专属四宫格可视化看板，保存至 figures/ 目录。"""
    sweep = list(report.get("sweep", ()))
    if not sweep:
        return {}

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = Path(figures_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    split = str(report.get("split", "test"))
    dataset = str(report.get("dataset", ""))
    stage = str(report.get("stage", ""))
    seed = report.get("seed", 0)

    etas = [float(item["eta"]) for item in sweep]
    x_labels = [
        f"η={e:g}\n(raw)" if abs(e) <= 1e-9 else f"η={e:g}" for e in etas
    ]
    x_pos = np.arange(len(etas))

    clean_50 = [
        float(item["clean_reference"].get("vocab50_macro_top10", 0.0)) * 100.0
        for item in sweep
    ]
    shift_50 = [
        float(item["shifted_aggregate"].get("mean_vocab50_macro_top10", 0.0))
        * 100.0
        for item in sweep
    ]
    gap_50 = [
        float(
            item["shifted_aggregate"].get(
                "clean_minus_shifted_vocab50_macro_top10", 0.0
            )
        )
        * 100.0
        for item in sweep
    ]
    chance_50 = (
        float(sweep[0]["clean_reference"].get("vocab50_chance_top10", 0.2))
        * 100.0
    )

    clean_fv = [
        float(item["clean_reference"].get("full_vocab_macro_top10", 0.0)) * 100.0
        for item in sweep
    ]
    shift_fv = [
        float(item["shifted_aggregate"].get("mean_full_vocab_macro_top10", 0.0))
        * 100.0
        for item in sweep
    ]
    clean_fv_mrr = [
        float(item["clean_reference"].get("full_vocab_mrr", 0.0)) * 100.0
        for item in sweep
    ]

    with evaluation_plot_context():
        fig, axes = plt.subplots(2, 2, figsize=(15, 10.5), dpi=160)
        fig.suptitle(
            f"Leave-One-Out Common-Mode Subtraction  |  {dataset}  |  stage={stage}  |  split={split}  |  seed={seed}",
            fontsize=13,
            fontweight="bold",
            y=0.98,
        )

        ax0 = axes[0, 0]
        width = 0.26
        bars_c = ax0.bar(
            x_pos - width,
            clean_50,
            width=width,
            label="Clean (0s) [-η·1/6 sum]",
            color=EVALUATION_PALETTE["clean"],
        )
        bars_s = ax0.bar(
            x_pos,
            shift_50,
            width=width,
            label="Shifted Mean [-η·1/5 LOO]",
            color=EVALUATION_PALETTE["shifted"],
        )
        bars_g = ax0.bar(
            x_pos + width,
            gap_50,
            width=width,
            label="Net Gap (Clean - Shifted)",
            color=EVALUATION_PALETTE["debiased"],
        )
        ax0.axhline(
            chance_50,
            color=EVALUATION_PALETTE["chance"],
            linestyle="--",
            linewidth=1.1,
            label=f"Chance ({chance_50:.1f}%)",
        )
        ax0.set_xticks(x_pos)
        ax0.set_xticklabels(x_labels)
        ax0.set_ylabel("Macro Top-10 Accuracy (%)")
        ax0.set_title(
            "A. Top-50 Vocabulary Retrieval Across Subtraction Weight η",
            fontweight="bold",
        )
        ax0.grid(axis="y", linestyle=":", alpha=0.5)
        ax0.legend(loc="best", fontsize=8.5)
        for b_group in (bars_c, bars_s, bars_g):
            for b in b_group:
                h = b.get_height()
                ax0.annotate(
                    f"{h:.1f}%",
                    xy=(b.get_x() + b.get_width() / 2.0, h),
                    xytext=(0, 3 if h >= 0 else -10),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=7.5,
                )

        ax1 = axes[0, 1]
        b_fv_c = ax1.bar(
            x_pos - width,
            clean_fv,
            width=width,
            label="Clean Full-Vocab Top-10",
            color=EVALUATION_PALETTE["clean"],
        )
        b_fv_s = ax1.bar(
            x_pos,
            shift_fv,
            width=width,
            label="Shifted Mean Full-Vocab Top-10",
            color=EVALUATION_PALETTE["shifted"],
        )
        b_fv_m = ax1.bar(
            x_pos + width,
            clean_fv_mrr,
            width=width,
            label="Clean Full-Vocab MRR",
            color="#9467bd",
        )
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(x_labels)
        ax1.set_ylabel("Percentage (%)")
        ax1.set_title(
            "B. Full-Vocabulary Retrieval & MRR Across Weight η",
            fontweight="bold",
        )
        ax1.grid(axis="y", linestyle=":", alpha=0.5)
        ax1.legend(loc="best", fontsize=8.5)
        for b_group in (b_fv_c, b_fv_s, b_fv_m):
            for b in b_group:
                h = b.get_height()
                ax1.annotate(
                    f"{h:.2f}%",
                    xy=(b.get_x() + b.get_width() / 2.0, h),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=7.5,
                )

        ax2 = axes[1, 0]
        cond_list = sweep[0].get("conditions", ())
        cond_names = ["Clean (0s)"] + [c["label"] for c in cond_list]
        cond_x = np.arange(len(cond_names))
        palette = ["#333333", "#1f77b4", "#2ca02c", "#d62728", "#9467bd"]
        for idx_e, item in enumerate(sweep):
            eta_val = float(item["eta"])
            vals = [
                float(item["clean_reference"].get("vocab50_macro_top10", 0.0))
                * 100.0
            ] + [
                float(c["summary"].get("vocab50_macro_top10", 0.0)) * 100.0
                for c in item.get("conditions", ())
            ]
            style = "--" if abs(eta_val) <= 1e-9 else "-"
            ax2.plot(
                cond_x,
                vals,
                marker="o",
                linestyle=style,
                linewidth=1.8,
                color=palette[idx_e % len(palette)],
                label=f"η={eta_val:g}"
                + (" (raw)" if abs(eta_val) <= 1e-9 else " (LOO)"),
            )
        ax2.axhline(
            chance_50,
            color="#888888",
            linestyle=":",
            linewidth=1.0,
            label="Chance",
        )
        ax2.set_xticks(cond_x)
        ax2.set_xticklabels(cond_names, rotation=25, ha="right")
        ax2.set_ylabel("Top-50 Macro Top-10 (%)")
        ax2.set_title(
            "C. Clean (0s) vs Individual Shifted Conditions (-120s..+120s)",
            fontweight="bold",
        )
        ax2.grid(True, linestyle=":", alpha=0.5)
        ax2.legend(loc="best", fontsize=8.5)

        ax3 = axes[1, 1]
        first_hz = sweep[0]["clean_reference"].get("horizons") or {}
        if first_hz and all(k in first_hz for k in ("K0", "K1", "K2")):
            hz_keys = ["K0", "K1", "K2"]
            hz_x = np.arange(len(hz_keys))
            for idx_e, item in enumerate(sweep):
                eta_val = float(item["eta"])
                c_hz = item["clean_reference"].get("horizons") or {}
                s_hz = item["shifted_aggregate"].get("horizons") or {}
                c_vals = [
                    float(
                        c_hz.get(k, {}).get("common_vocab50_macro_top10", 0.0)
                    )
                    * 100.0
                    for k in hz_keys
                ]
                s_vals = [
                    float(
                        s_hz.get(k, {}).get(
                            "mean_common_vocab50_macro_top10", 0.0
                        )
                    )
                    * 100.0
                    for k in hz_keys
                ]
                col = palette[idx_e % len(palette)]
                ax3.plot(
                    hz_x,
                    c_vals,
                    marker="o",
                    linestyle="-",
                    color=col,
                    label=f"Clean η={eta_val:g}",
                )
                ax3.plot(
                    hz_x,
                    s_vals,
                    marker="s",
                    linestyle=":",
                    color=col,
                    alpha=0.75,
                    label=f"Shifted η={eta_val:g}",
                )
            ax3.set_xticks(hz_x)
            ax3.set_xticklabels(["K0 (+0w)", "K1 (+1w)", "K2 (+2w)"])
            ax3.set_ylabel("Common Top-50 Macro Top-10 (%)")
            ax3.set_title(
                "D. Rolling Revision Horizons (K0 → K1 → K2) Across η",
                fontweight="bold",
            )
            ax3.grid(True, linestyle=":", alpha=0.5)
            ax3.legend(loc="best", fontsize=7.5, ncol=2)
        else:
            med_clean = [
                float(item["clean_reference"].get("full_vocab_median_rank", 0.0))
                for item in sweep
            ]
            med_shift = [
                float(
                    item["shifted_aggregate"].get(
                        "mean_full_vocab_median_rank", 0.0
                    )
                )
                for item in sweep
            ]
            b_mc = ax3.bar(
                x_pos - width / 2.0,
                med_clean,
                width=width,
                label="Clean Full-Vocab Median Rank (↓ better)",
                color=EVALUATION_PALETTE["clean"],
            )
            b_ms = ax3.bar(
                x_pos + width / 2.0,
                med_shift,
                width=width,
                label="Shifted Mean Median Rank",
                color=EVALUATION_PALETTE["shifted"],
            )
            ax3.set_xticks(x_pos)
            ax3.set_xticklabels(x_labels)
            ax3.set_ylabel("Median Rank (Lower is Better)")
            ax3.set_title(
                "D. Full-Vocabulary Median Rank Across Weight η",
                fontweight="bold",
            )
            ax3.grid(axis="y", linestyle=":", alpha=0.5)
            ax3.legend(loc="best", fontsize=8.5)
            for b_group in (b_mc, b_ms):
                for b in b_group:
                    h = b.get_height()
                    ax3.annotate(
                        f"{h:.0f}",
                        xy=(b.get_x() + b.get_width() / 2.0, h),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha="center",
                        va="bottom",
                        fontsize=7.5,
                    )

        fig.tight_layout(rect=(0, 0, 1, 0.95))

    png_path = out_dir / f"common_mode_subtraction_{split}.png"
    pdf_path = out_dir / f"common_mode_subtraction_{split}.pdf"
    _save_dual_format(fig, png_path, pdf_path, dpi=160, bbox_inches="tight")
    return {
        "common_mode_subtraction_png": str(png_path),
        "common_mode_subtraction_pdf": str(pdf_path),
    }


def plot_subspace_coherence_debiasing_figures(
    report: Mapping[str, Any],
    figures_dir: str | Path,
) -> dict[str, str]:
    """生成方案一（低秩子空间投影 + 严格留一法基底 LOO 去偏）专属四宫格诊断看板图。"""
    subspace_entries = list(report.get("subspace_sweep", ()))
    if not subspace_entries:
        return {}

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = Path(figures_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    split = str(report.get("split", "test"))
    dataset = str(report.get("dataset", ""))
    stage = str(report.get("stage", ""))
    seed = report.get("seed", 0)
    raw_entry = report["raw_baseline"]
    full_entries = list(report.get("full_rank_loo_baselines", ()))

    with evaluation_plot_context():
        fig, axes = plt.subplots(2, 2, figsize=(15.6, 11.0), dpi=160)
        fig.suptitle(
            f"Scheme 1: Subspace-Projected LOO Debiasing (Strict LOO Basis U_{{K,LOO}})  |  {dataset}  |  stage={stage}  |  split={split}  |  seed={seed}",
            fontsize=13,
            fontweight="bold",
            y=0.98,
        )

        # Panel A: Singular Value Cumulative Energy Explained across Rank K
        ax0 = axes[0, 0]
        diags = report.get("subspace_and_coherence_diagnostics") or {}
        rank_ticks = [1, 2, 4, 8, 12, 16, 24, 32, 48, 64]
        colors_h = {
            "K0": EVALUATION_PALETTE["K0"],
            "K1": EVALUATION_PALETTE["K1"],
            "K2": EVALUATION_PALETTE["K2"],
            "main": EVALUATION_PALETTE["clean"],
        }
        for h_key, d_item in diags.items():
            cum = d_item.get("cumulative_energy_explained", {})
            xs = [r for r in rank_ticks if f"top_{r}" in cum]
            ys = [float(cum[f"top_{r}"]) * 100.0 for r in xs]
            if xs:
                eff_r = float(
                    d_item.get("effective_rank_participation_ratio", 0.0)
                )
                ax0.plot(
                    xs,
                    ys,
                    "-o",
                    linewidth=2.0,
                    color=colors_h.get(h_key, "#1f77b4"),
                    label=f"{h_key} (EffRank={eff_r:.1f})",
                )
        ax0.axvline(
            8,
            color=EVALUATION_PALETTE["highlight"],
            linestyle="--",
            linewidth=1.2,
            label="Recommended Rank K=8..12",
        )
        ax0.set_xlabel("Subspace Rank K (Centered SVD Singular Vectors)")
        ax0.set_ylabel("Cumulative Shift Bias Variance Explained (%)")
        ax0.set_title(
            "A. Shift Common-Mode Centered SVD Energy Spectrum",
            fontweight="bold",
        )
        ax0.grid(True, linestyle=":", alpha=0.5)
        ax0.legend(loc="lower right", fontsize=8.5)

        # Panel B: Clean vs Shifted Mean vs Net Gap Across Rank K
        ax1 = axes[0, 1]
        gate_off_sub = [
            e
            for e in subspace_entries
            if (not e.get("use_coherence_gate"))
            and abs(float(e["eta"]) - 0.8) <= 1e-6
        ]
        if not gate_off_sub:
            target_eta = float(subspace_entries[0]["eta"])
            gate_off_sub = [
                e
                for e in subspace_entries
                if (not e.get("use_coherence_gate"))
                and abs(float(e["eta"]) - target_eta) <= 1e-6
            ]
            if not gate_off_sub:
                gate_off_sub = [
                    e
                    for e in subspace_entries
                    if abs(float(e["eta"]) - target_eta) <= 1e-6
                ]

        full_eta1 = next(
            (e for e in full_entries if abs(float(e["eta"]) - 1.0) <= 1e-6),
            full_entries[-1] if full_entries else raw_entry,
        )
        compare_items = [raw_entry, *gate_off_sub, full_eta1]
        x_labels = [
            "Raw\n(η=0)"
            if item["method_family"] == "raw_baseline"
            else (
                f"Full-Rank\n(D, η={item['eta']:g})"
                if item["method_family"] == "full_rank_loo"
                else f"Rank K={item['rank_k']}\n(η={item['eta']:g}, Gate={'On' if item.get('use_coherence_gate') else 'Off'})"
            )
            for item in compare_items
        ]
        x_pos = np.arange(len(compare_items))
        c_vals = [
            float(item["clean_reference"].get("vocab50_macro_top10", 0.0)) * 100.0
            for item in compare_items
        ]
        s_vals = [
            float(item["shifted_aggregate"].get("mean_vocab50_macro_top10", 0.0))
            * 100.0
            for item in compare_items
        ]
        g_vals = [
            float(
                item["shifted_aggregate"].get(
                    "clean_minus_shifted_vocab50_macro_top10", 0.0
                )
            )
            * 100.0
            for item in compare_items
        ]
        width = 0.26
        bars_c = ax1.bar(
            x_pos - width,
            c_vals,
            width=width,
            label="Clean (0s)",
            color=EVALUATION_PALETTE["clean"],
        )
        bars_s = ax1.bar(
            x_pos,
            s_vals,
            width=width,
            label="Shifted Mean (Strict LOO)",
            color=EVALUATION_PALETTE["shifted"],
        )
        bars_g = ax1.bar(
            x_pos + width,
            g_vals,
            width=width,
            label="Net Gap (Clean - Shift)",
            color=EVALUATION_PALETTE["debiased"],
        )

        ctx_ref = report.get("context_stage_reference")
        if (
            isinstance(ctx_ref, Mapping)
            and ctx_ref.get("vocab50_macro_top10") is not None
        ):
            ctx_50 = float(ctx_ref["vocab50_macro_top10"]) * 100.0
            ax1.axhline(
                ctx_50,
                color="#9467bd",
                linestyle="-.",
                linewidth=1.4,
                label=f"Stage 2.1 Context ({ctx_50:.2f}%)",
            )
        chance_50 = (
            float(raw_entry["clean_reference"].get("vocab50_chance_top10", 0.2))
            * 100.0
        )
        ax1.axhline(
            chance_50,
            color=EVALUATION_PALETTE["chance"],
            linestyle="--",
            linewidth=1.0,
            label=f"Chance ({chance_50:.1f}%)",
        )
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(x_labels, fontsize=8.5)
        ax1.set_ylabel("50-Word Macro Top-10 (%)")
        ax1.set_title(
            "B. 50-Word Macro Top-10: Subspace Rank K vs Full-Rank LOO",
            fontweight="bold",
        )
        ax1.grid(axis="y", linestyle=":", alpha=0.5)
        ax1.legend(loc="best", fontsize=8.0)
        for b_group in (bars_c, bars_s, bars_g):
            for b in b_group:
                h = b.get_height()
                ax1.annotate(
                    f"{h:.1f}%",
                    xy=(b.get_x() + b.get_width() / 2.0, h),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=7.2,
                )

        # Panel C: Revision Horizons (K0 / K1 / K2) Clean vs Shifted Mean
        ax2 = axes[1, 0]
        best_sub = (
            report.get("best_by_clean_preservation_vocab50") or subspace_entries[0]
        )
        raw_hz = raw_entry["clean_reference"].get("horizons") or {}
        if raw_hz and all(k in raw_hz for k in ("K0", "K1", "K2")):
            hz_keys = ["K0", "K1", "K2"]
            hz_x = np.arange(len(hz_keys))
            series = [
                ("Raw (η=0)", raw_entry, "#333333"),
                (
                    f"Full-Rank LOO (η={full_eta1['eta']:g})",
                    full_eta1,
                    EVALUATION_PALETTE["highlight"],
                ),
                (
                    f"Scheme 1 ({best_sub['label']})",
                    best_sub,
                    EVALUATION_PALETTE["debiased"],
                ),
            ]
            for s_label, item, col in series:
                c_hz = item["clean_reference"].get("horizons") or {}
                s_hz = item["shifted_aggregate"].get("horizons") or {}
                c_line = [
                    float(
                        c_hz.get(k, {}).get("common_vocab50_macro_top10", 0.0)
                    )
                    * 100.0
                    for k in hz_keys
                ]
                s_line = [
                    float(
                        s_hz.get(k, {}).get(
                            "mean_common_vocab50_macro_top10", 0.0
                        )
                    )
                    * 100.0
                    for k in hz_keys
                ]
                ax2.plot(
                    hz_x,
                    c_line,
                    "-o",
                    linewidth=2.0,
                    color=col,
                    label=f"Clean: {s_label}",
                )
                ax2.plot(
                    hz_x,
                    s_line,
                    ":s",
                    linewidth=1.8,
                    color=col,
                    label=f"Shifted: {s_label}",
                )
            ax2.axhline(chance_50, color="#888888", linestyle="--", linewidth=1.0)
            ax2.set_xticks(hz_x)
            ax2.set_xticklabels(["K0 (0w)", "K1 (+1w)", "K2 (+2w)"])
            ax2.set_ylabel("Common 50-Word Macro Top-10 (%)")
            ax2.set_title(
                "C. Horizons K0/K1/K2: Preserving Clean While Suppressing Shift",
                fontweight="bold",
            )
            ax2.grid(True, linestyle=":", alpha=0.5)
            ax2.legend(loc="best", fontsize=7.8)
        else:
            for rank_k in report.get("grid", {}).get("rank_k_values", ()):
                sub_k = [
                    e
                    for e in subspace_entries
                    if e.get("rank_k") == rank_k
                    and (not e.get("use_coherence_gate"))
                ]
                if sub_k:
                    etas_k = [float(e["eta"]) for e in sub_k]
                    gaps_k = [
                        float(
                            e["shifted_aggregate"].get(
                                "clean_minus_shifted_vocab50_macro_top10", 0.0
                            )
                        )
                        * 100.0
                        for e in sub_k
                    ]
                    ax2.plot(
                        etas_k,
                        gaps_k,
                        "-o",
                        linewidth=2.0,
                        label=f"Rank K={rank_k} (Gate=Off)",
                    )
            ax2.set_xlabel("Subtraction Weight η")
            ax2.set_ylabel("Net Gap: Clean - Shifted Mean (%)")
            ax2.set_title(
                "C. Net Gap Across η and Subspace Rank K", fontweight="bold"
            )
            ax2.grid(True, linestyle=":", alpha=0.5)
            ax2.legend(loc="best", fontsize=8.5)

        # Panel D: Per-Shift Condition Profile
        ax3 = axes[1, 1]
        cond_list = raw_entry.get("conditions", ())
        cond_names = ["Clean (0s)"] + [c["label"] for c in cond_list]
        cond_x = np.arange(len(cond_names))
        profile_series = [
            ("Raw Baseline (η=0)", raw_entry, "#333333", "--"),
            (
                f"Full-Rank LOO (η={full_eta1['eta']:g})",
                full_eta1,
                EVALUATION_PALETTE["highlight"],
                "-.",
            ),
            (best_sub["label"], best_sub, EVALUATION_PALETTE["debiased"], "-"),
        ]
        for p_label, item, col, ls in profile_series:
            vals = [
                float(item["clean_reference"].get("vocab50_macro_top10", 0.0))
                * 100.0
            ] + [
                float(c["summary"].get("vocab50_macro_top10", 0.0)) * 100.0
                for c in item.get("conditions", ())
            ]
            ax3.plot(
                cond_x,
                vals,
                marker="o",
                linestyle=ls,
                linewidth=2.0,
                color=col,
                label=p_label,
            )
        ax3.axhline(
            chance_50,
            color="#888888",
            linestyle=":",
            linewidth=1.0,
            label="Chance (20.0%)",
        )
        ax3.set_xticks(cond_x)
        ax3.set_xticklabels(cond_names, rotation=25, ha="right")
        ax3.set_ylabel("50-Word Macro Top-10 (%)")
        ax3.set_title(
            "D. Clean (0s) vs Historical Shifted Conditions (-15s..-120s)",
            fontweight="bold",
        )
        ax3.grid(True, linestyle=":", alpha=0.5)
        ax3.legend(loc="best", fontsize=8.0)

        fig.tight_layout(rect=[0, 0.01, 1, 0.95])

    png_path = out_dir / f"subspace_coherence_debiasing_{split}.png"
    pdf_path = out_dir / f"subspace_coherence_debiasing_{split}.pdf"
    _save_dual_format(fig, png_path, pdf_path, dpi=160)
    return {
        "subspace_coherence_debiasing_png": str(png_path),
        "subspace_coherence_debiasing_pdf": str(pdf_path),
    }


def plot_confidence_gate_pareto_curve(
    pareto_result: Mapping[str, Any],
    output_dir: str | Path,
    *,
    dataset: str = "pallier2025",
    split: str = "test",
) -> dict[str, str]:
    """绘制 C-gated 置信度门控“解码质量 vs 屏幕改写率”帕累托曲线与闪烁构成图。"""
    curve = pareto_result.get("pareto_curve") or []
    if not curve:
        return {}

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rec_pt = pareto_result["recommended_operating_point"]

    taus = [pt["tau_gate"] for pt in curve]
    rewrite_rates = [pt["screen_rewrite_rate_pct"] for pt in curve]
    bleu1_vals = [pt["bleu1_pct"] for pt in curve]
    word_acc_vals = [pt["word_acc_pct"] for pt in curve]
    bert_vals = [pt["bertscore_f1_pct"] for pt in curve]
    flicker_rates = [pt["invalid_flicker_rate_pct"] for pt in curve]
    repair_rates = [pt["beneficial_repair_rate_pct"] for pt in curve]
    harm_rates = [pt["harmful_rewrite_rate_pct"] for pt in curve]

    with evaluation_plot_context():
        fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(13.5, 5.2), dpi=160)
        fig.suptitle(
            f"S3 Confidence-Gated Revision (C-gated) Pareto Analysis | {dataset} ({split.upper()})",
            fontsize=12.5,
            fontweight="bold",
        )

        ax_left.plot(
            rewrite_rates,
            bleu1_vals,
            "-o",
            color=EVALUATION_PALETTE["clean"],
            linewidth=2.2,
            markersize=5,
            label="BLEU-1 (%)",
        )
        ax_left.plot(
            rewrite_rates,
            word_acc_vals,
            "-s",
            color=EVALUATION_PALETTE["debiased"],
            linewidth=2.2,
            markersize=5,
            label="Word Acc (%)",
        )
        ax_left.plot(
            rewrite_rates,
            bert_vals,
            "-^",
            color="#9467bd",
            linewidth=1.8,
            markersize=4.5,
            alpha=0.85,
            label="BERTScore F1 (%)",
        )
        ax_left.axvline(
            rec_pt["screen_rewrite_rate_pct"],
            color=EVALUATION_PALETTE["highlight"],
            linestyle="--",
            linewidth=1.4,
            label=f"Recommended τ={rec_pt['tau_gate']:.1f} (Rewrite={rec_pt['screen_rewrite_rate_pct']:.1f}%)",
        )
        for idx in (0, len(curve) // 2, len(curve) - 1):
            pt = curve[idx]
            ax_left.annotate(
                f"τ={pt['tau_gate']:.1f}",
                xy=(pt["screen_rewrite_rate_pct"], pt["bleu1_pct"]),
                xytext=(4, 6),
                textcoords="offset points",
                fontsize=8,
            )

        ax_left.set_title(
            "(A) Decoding Quality vs. Screen Rewrite Rate (Pareto Curve)",
            fontsize=10.5,
            fontweight="bold",
        )
        ax_left.set_xlabel("Screen Rewrite / Flicker Rate (%)", fontsize=10)
        ax_left.set_ylabel("Decoding Quality (%)", fontsize=10)
        ax_left.grid(True, linestyle=":", alpha=0.5)
        ax_left.legend(loc="best", fontsize=8.5)

        ax_right.plot(
            taus,
            rewrite_rates,
            "-o",
            color="#333333",
            linewidth=2.0,
            markersize=4.5,
            label="Total Screen Rewrite Rate (%)",
        )
        ax_right.plot(
            taus,
            flicker_rates,
            "-d",
            color=EVALUATION_PALETTE["shifted"],
            linewidth=2.0,
            markersize=4.5,
            label="Invalid Flicker (Wrong→Wrong %)",
        )
        ax_right.plot(
            taus,
            repair_rates,
            "-s",
            color=EVALUATION_PALETTE["debiased"],
            linewidth=2.0,
            markersize=4.5,
            label="Beneficial Repair (Wrong→Right %)",
        )
        ax_right.plot(
            taus,
            harm_rates,
            "-x",
            color=EVALUATION_PALETTE["highlight"],
            linewidth=1.8,
            markersize=4.5,
            label="Harmful Rewrite (Right→Wrong %)",
        )
        ax_right.axvline(
            rec_pt["tau_gate"],
            color=EVALUATION_PALETTE["highlight"],
            linestyle="--",
            linewidth=1.3,
        )

        ax_right.set_title(
            "(B) Screen Rewrite Decomposition Across Gate Threshold τ_gate",
            fontsize=10.5,
            fontweight="bold",
        )
        ax_right.set_xlabel("Confidence Margin Threshold τ_gate", fontsize=10)
        ax_right.set_ylabel("Rate per Revision Opportunity (%)", fontsize=10)
        ax_right.grid(True, linestyle=":", alpha=0.5)
        ax_right.legend(loc="best", fontsize=8.5)

        fig.tight_layout(rect=[0, 0, 1, 0.94])

    png_path = out_dir / f"pareto_confidence_gate_{split}.png"
    pdf_path = out_dir / f"pareto_confidence_gate_{split}.pdf"
    _save_dual_format(fig, png_path, pdf_path, dpi=160)
    return {
        "pareto_png": str(png_path),
        "pareto_pdf": str(pdf_path),
    }


def plot_baseline_and_shifted_comparison(
    condition_metrics: Mapping[str, Mapping[str, float | int]],
    output_dir: str | Path,
    *,
    dataset: str = "pallier2025",
    split: str = "test",
) -> dict[str, str]:
    """绘制 4 档输入延迟预算（S0, K0, K1, S3_clean, F）、LM-Only 及 Shifted (-75s) 的 6 维指标对比图。"""
    filtered_labels = [
        k
        for k, v in condition_metrics.items()
        if str(k) != "F_local"
        and not str(k).endswith("_c_gated")
        and not (isinstance(v, Mapping) and bool(v.get("is_online_screen_gate")))
    ]
    labels = filtered_labels if filtered_labels else list(condition_metrics.keys())
    if not labels:
        return {}

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    x = np.arange(len(labels))
    width = 0.18

    with evaluation_plot_context():
        fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(14.0, 5.2), dpi=160)
        fig.suptitle(
            f"S3 Latency-Budget Tiers & Shifted (-75s) Control | {dataset} ({split.upper()})",
            fontsize=12.5,
            fontweight="bold",
        )

        bleu1 = [float(condition_metrics[k].get("bleu1_pct", 0.0)) for k in labels]
        word_acc = [
            float(condition_metrics[k].get("word_acc_pct", 0.0)) for k in labels
        ]
        content_acc = [
            float(condition_metrics[k].get("content_word_acc_pct", 0.0))
            for k in labels
        ]
        function_acc = [
            float(condition_metrics[k].get("function_word_acc_pct", 0.0))
            for k in labels
        ]

        ax_left.bar(
            x - 1.5 * width,
            bleu1,
            width,
            label="BLEU-1 (%)",
            color=EVALUATION_PALETTE["clean"],
        )
        ax_left.bar(
            x - 0.5 * width,
            word_acc,
            width,
            label="Word Acc (%)",
            color=EVALUATION_PALETTE["debiased"],
        )
        ax_left.bar(
            x + 0.5 * width,
            content_acc,
            width,
            label="Content Word Acc (%)",
            color=EVALUATION_PALETTE["shifted"],
        )
        ax_left.bar(
            x + 1.5 * width,
            function_acc,
            width,
            label="Function Word Acc (%)",
            color="#9467bd",
        )
        ax_left.set_title(
            "(A) Lexical & Content/Function Word Accuracy",
            fontsize=10.5,
            fontweight="bold",
        )
        ax_left.set_xticks(x)
        ax_left.set_xticklabels(labels, rotation=18, ha="right", fontsize=9)
        ax_left.set_ylabel("Accuracy / Score (%)", fontsize=10)
        ax_left.grid(axis="y", linestyle=":", alpha=0.5)
        ax_left.legend(loc="best", fontsize=8.5)

        bert_f1 = [
            float(condition_metrics[k].get("bertscore_f1_pct", 0.0))
            for k in labels
        ]
        wer_vals = [
            float(condition_metrics[k].get("wer_pct", 0.0)) for k in labels
        ]
        cer_vals = [
            float(condition_metrics[k].get("cer_pct", 0.0)) for k in labels
        ]

        w2 = 0.24
        ax_right.bar(
            x - w2, bert_f1, w2, label="BERTScore F1 (%) ↑", color="#17becf"
        )
        ax_right.bar(
            x,
            wer_vals,
            w2,
            label="WER (%) ↓",
            color=EVALUATION_PALETTE["highlight"],
            alpha=0.85,
        )
        ax_right.bar(
            x + w2, cer_vals, w2, label="CER (%) ↓", color="#8c564b", alpha=0.85
        )
        ax_right.set_title(
            "(B) Semantic BERTScore & Error Rates (WER / CER)",
            fontsize=10.5,
            fontweight="bold",
        )
        ax_right.set_xticks(x)
        ax_right.set_xticklabels(labels, rotation=18, ha="right", fontsize=9)
        ax_right.set_ylabel("Percentage (%)", fontsize=10)
        ax_right.grid(axis="y", linestyle=":", alpha=0.5)
        ax_right.legend(loc="best", fontsize=8.5)

        fig.tight_layout(rect=[0, 0, 1, 0.94])

    png_path = out_dir / f"baseline_and_shifted_comparison_{split}.png"
    pdf_path = out_dir / f"baseline_and_shifted_comparison_{split}.pdf"
    _save_dual_format(fig, png_path, pdf_path, dpi=160)
    return {
        "comparison_png": str(png_path),
        "comparison_pdf": str(pdf_path),
    }


def plot_multi_subject_observation_curve(
    multi_subject_scaling: Mapping[str, Mapping[str, Any]],
    output_dir: str | Path,
    *,
    dataset: str = "pallier2025",
    split: str = "test",
) -> dict[str, str]:
    """绘制跨被试观测数 k ∈ {1, 2, 5, 10} 聚合下的序列解码性能跃升曲线（含核心词表子集轨迹）。"""
    if not multi_subject_scaling:
        return {}

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ordered_items = sorted(
        multi_subject_scaling.items(),
        key=lambda kv: int(kv[1].get("k_observations", 1)),
    )
    k_vals = [int(item[1].get("k_observations", 1)) for item in ordered_items]
    bleu1 = [float(item[1].get("bleu1_pct", 0.0)) for item in ordered_items]
    corpus_bleu1 = [
        float(item[1].get("corpus_bleu1_pct", 0.0)) for item in ordered_items
    ]
    oracle_bleu1 = [
        float(item[1].get("beam_oracle_bleu1_pct", 0.0))
        for item in ordered_items
    ]
    word_acc = [
        float(item[1].get("word_acc_pct", 0.0)) for item in ordered_items
    ]
    invocab_acc = [
        float(item[1].get("in_vocabulary_word_acc_pct", 0.0))
        for item in ordered_items
    ]
    top5_acc = [
        float(item[1].get("word_acc_top5_pct", 0.0)) for item in ordered_items
    ]

    with evaluation_plot_context():
        fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(13.5, 5.0), dpi=160)
        fig.suptitle(
            f"Multi-Subject Observation Scaling (k = {', '.join(map(str, k_vals))}) | {dataset} ({split.upper()})",
            fontsize=12.5,
            fontweight="bold",
        )

        ax_left.plot(
            k_vals,
            bleu1,
            "-o",
            color=EVALUATION_PALETTE["clean"],
            linewidth=2.2,
            markersize=6,
            label="Full Vocab BLEU-1 (%)",
        )
        ax_left.plot(
            k_vals,
            corpus_bleu1,
            "-s",
            color=EVALUATION_PALETTE["debiased"],
            linewidth=2.2,
            markersize=6,
            label="Corpus BLEU-1 (%)",
        )
        ax_left.plot(
            k_vals,
            oracle_bleu1,
            "--^",
            color="#9467bd",
            linewidth=2.0,
            markersize=6,
            label="Beam-5 Oracle BLEU-1 (%)",
        )

        ax_right.plot(
            k_vals,
            word_acc,
            "-o",
            color=EVALUATION_PALETTE["shifted"],
            linewidth=2.2,
            markersize=6,
            label="Full Vocab Word Acc@1 (%)",
        )
        ax_right.plot(
            k_vals,
            invocab_acc,
            "-s",
            color=EVALUATION_PALETTE["highlight"],
            linewidth=2.2,
            markersize=6,
            label="In-Vocabulary Word Acc@1 (%)",
        )
        ax_right.plot(
            k_vals,
            top5_acc,
            "--d",
            color="#17becf",
            linewidth=2.0,
            markersize=6,
            label="Position Candidate Acc@5 (%)",
        )

        core_colors = {"V=50": "#2ca02c", "V=100": "#ff7f0e", "V=250": "#d62728"}
        for vk, v_col in core_colors.items():
            v_c80_bleu: list[float] = []
            v_c80_acc: list[float] = []
            valid_v = True
            for _, item_val in ordered_items:
                by_v = item_val.get("by_vocabulary_size") if isinstance(item_val, Mapping) else None
                v_blk = by_v.get(vk) if isinstance(by_v, Mapping) else None
                c80 = v_blk.get("core_subset_80pct") if isinstance(v_blk, Mapping) else None
                if not isinstance(c80, Mapping) or "bleu1_pct" not in c80:
                    valid_v = False
                    break
                v_c80_bleu.append(float(c80.get("bleu1_pct", 0.0)))
                v_c80_acc.append(float(c80.get("word_acc_pct", 0.0)))
            if valid_v and v_c80_bleu:
                ax_left.plot(
                    k_vals,
                    v_c80_bleu,
                    ":D",
                    color=v_col,
                    linewidth=1.8,
                    markersize=5,
                    label=f"{vk} (>=80% Core) BLEU-1",
                )
                ax_right.plot(
                    k_vals,
                    v_c80_acc,
                    ":D",
                    color=v_col,
                    linewidth=1.8,
                    markersize=5,
                    label=f"{vk} (>=80% Core) Word Acc",
                )

        ax_left.set_title(
            "(A) Sequence BLEU-1 vs. Aggregated Subject Observations k",
            fontsize=10.5,
            fontweight="bold",
        )
        ax_left.set_xlabel("Number of Averaged Subject Observations (k)", fontsize=10)
        ax_left.set_ylabel("BLEU-1 (%)", fontsize=10)
        ax_left.set_xticks(k_vals)
        ax_left.grid(True, linestyle=":", alpha=0.5)
        ax_left.legend(loc="best", fontsize=8.0)

        ax_right.set_title(
            "(B) Word Accuracy (Top-1 / In-Vocab / Core) vs. Observations k",
            fontsize=10.5,
            fontweight="bold",
        )
        ax_right.set_xlabel("Number of Averaged Subject Observations (k)", fontsize=10)
        ax_right.set_ylabel("Word Accuracy (%)", fontsize=10)
        ax_right.set_xticks(k_vals)
        ax_right.grid(True, linestyle=":", alpha=0.5)
        ax_right.legend(loc="best", fontsize=8.0)

        fig.tight_layout(rect=[0, 0, 1, 0.94])

    png_path = out_dir / f"multi_subject_observation_scaling_{split}.png"
    pdf_path = out_dir / f"multi_subject_observation_scaling_{split}.pdf"
    _save_dual_format(fig, png_path, pdf_path, dpi=160)
    return {
        "multi_subject_scaling_png": str(png_path),
        "multi_subject_scaling_pdf": str(pdf_path),
    }


def plot_vocabulary_scaling_curve(
    vocabulary_scaling_summary: Mapping[str, Mapping[str, Any]],
    output_dir: str | Path,
    *,
    dataset: str = "pallier2025",
    split: str = "test",
) -> dict[str, str]:
    """绘制 1x4 多尺度词表敏感性曲线（横轴 log |V|，纵轴 BLEU-1 / In-Vocab Word Acc / CER / BERTScore）。"""
    if not vocabulary_scaling_summary:
        return {}

    ordered = sorted(
        vocabulary_scaling_summary.items(),
        key=lambda kv: int(kv[1].get("vocabulary_size", 0)),
    )
    v_sizes = [int(item[1].get("vocabulary_size", 0)) for item in ordered]
    if len(v_sizes) < 2:
        return {}

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    curve_specs = (
        (
            "S3_clean",
            "S3 (K2 Rolling Revision, Clean)",
            EVALUATION_PALETTE["clean"],
            "-o",
            2.3,
        ),
        (
            "K0",
            "K0 (Initial Draft, Clean)",
            EVALUATION_PALETTE["debiased"],
            "--s",
            1.9,
        ),
        (
            "S0",
            "S0 (Single-Word CNN, Clean)",
            EVALUATION_PALETTE["shifted"],
            ":^",
            1.8,
        ),
        (
            "S3_shifted_neg75s",
            "S3 Shifted (-75s Control)",
            EVALUATION_PALETTE["highlight"],
            "-.x",
            1.7,
        ),
    )
    panel_specs = (
        ("bleu1_pct", "(A) Sentence BLEU-1 (%)", "BLEU-1 (%)"),
        (
            "in_vocabulary_word_acc_pct",
            "(B) In-Vocabulary Word Acc (%)",
            "In-Vocab Word Acc (%)",
        ),
        ("cer_pct", "(C) Character Error Rate CER (%) ↓", "CER (%)"),
        (
            "bertscore_f1_pct",
            "(D) Continuous Semantic BERTScore (%)",
            "BERTScore F1 (%)",
        ),
    )

    with evaluation_plot_context():
        fig, axes = plt.subplots(1, 4, figsize=(18.5, 4.6), dpi=160)
        fig.suptitle(
            f"Multi-Scale Vocabulary Scaling Law (|V| = {', '.join(map(str, v_sizes))}) | {dataset} ({split.upper()})",
            fontsize=12.5,
            fontweight="bold",
        )

        for ax, (metric_key, panel_title, y_label) in zip(axes, panel_specs):
            for cond_key, label, color, style, lw in curve_specs:
                xs: list[int] = []
                ys: list[float] = []
                for _, scale_obj in ordered:
                    c_map = scale_obj.get("conditions") or {}
                    if cond_key in c_map and metric_key in c_map[cond_key]:
                        xs.append(int(scale_obj["vocabulary_size"]))
                        ys.append(float(c_map[cond_key][metric_key]))
                if xs:
                    ax.plot(
                        xs,
                        ys,
                        style,
                        color=color,
                        linewidth=lw,
                        markersize=5.5,
                        label=label,
                    )
            ax.set_xscale("log")
            ax.set_xticks(v_sizes)
            ax.set_xticklabels([str(v) for v in v_sizes], fontsize=8.5)
            ax.get_xaxis().set_minor_formatter(matplotlib.ticker.NullFormatter())
            ax.set_title(panel_title, fontsize=10.0, fontweight="bold")
            ax.set_xlabel("Candidate Vocabulary Size |V| (log scale)", fontsize=9.5)
            ax.set_ylabel(y_label, fontsize=9.5)
            ax.grid(True, linestyle=":", alpha=0.5)

        handles, labels = axes[0].get_legend_handles_labels()
        if handles:
            fig.legend(
                handles,
                labels,
                loc="lower center",
                ncol=4,
                fontsize=9.0,
                frameon=True,
                bbox_to_anchor=(0.5, 0.01),
            )

        fig.tight_layout(rect=[0, 0.08, 1, 0.92])

    png_path = out_dir / f"s3_vocabulary_scaling_{split}.png"
    pdf_path = out_dir / f"s3_vocabulary_scaling_{split}.pdf"
    _save_dual_format(fig, png_path, pdf_path, dpi=160)
    return {
        "vocabulary_scaling_png": str(png_path),
        "vocabulary_scaling_pdf": str(pdf_path),
    }


def replot_from_saved_json(
    input_path: str | Path,
    output_dir: str | Path | None = None,
) -> dict[str, str]:
    """从任意已保存评估 JSON（或实验运行目录）零成本重新生成全套 PNG/PDF 可视化图表。"""
    target = Path(input_path).resolve()
    if target.is_dir():
        candidates = [
            target / "multi_stage_evaluation_test.json",
            target / "multi_stage_evaluation_val.json",
            target / "evaluation_test.json",
            target / "evaluation_val.json",
            target / "s3_evaluation_summary_test.json",
            target / "s3_evaluation_summary_val.json",
            target / "exploratory" / "subspace_coherence_debiasing_test.json",
            target / "exploratory" / "common_mode_subtraction_test.json",
        ]
        found = next((c for c in candidates if c.is_file()), None)
        if found is None:
            raise FileNotFoundError(f"在目录 {target} 下未找到任何已保存的评估 JSON 文件。")
        target = found
    elif not target.is_file():
        raise FileNotFoundError(f"指定的评估 JSON 文件不存在：{target}")

    payload = json.loads(target.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON 文件格式非法（期望顶层对象为 dict）：{target}")

    default_fig_dir = (
        target.parent
        if payload.get("protocol") == "sequence_decoding_s3_evaluation"
        else (
            target.parent.parent / "figures"
            if target.parent.name == "exploratory"
            else target.parent / "figures"
        )
    )
    fig_dir = Path(output_dir).resolve() if output_dir is not None else default_fig_dir

    status = str(payload.get("status", ""))
    exp_type = str(payload.get("experiment_type", ""))
    protocol = str(payload.get("protocol", ""))
    split = str(payload.get("split", "test"))
    dataset = str(payload.get("dataset", "pallier2025"))

    if status == "multi_stage_evaluation_completed" or "stages" in payload:
        return plot_stage_comparison_figure(
            payload["stages"], fig_dir, split=split
        )
    if exp_type == "exploratory_leave_one_out_common_mode_subtraction":
        return plot_common_mode_subtraction_figures(payload, fig_dir)
    if exp_type == "exploratory_subspace_coherence_loo_debiasing":
        return plot_subspace_coherence_debiasing_figures(payload, fig_dir)
    if protocol == "sequence_decoding_s3_evaluation":
        generated: dict[str, str] = {}
        conds = payload.get("conditions") or {}
        if conds:
            generated.update(
                plot_baseline_and_shifted_comparison(
                    conds, fig_dir, dataset=dataset, split=split
                )
            )
        pareto_map = payload.get("confidence_gate_pareto") or {}
        pareto_main = pareto_map.get("S3_clean") or pareto_map.get("S3")
        if isinstance(pareto_main, Mapping) and pareto_main.get("pareto_curve"):
            generated.update(
                plot_confidence_gate_pareto_curve(
                    pareto_main, fig_dir, dataset=dataset, split=split
                )
            )
        ms_scaling = payload.get("multi_subject_scaling") or {}
        if len(ms_scaling) >= 2:
            generated.update(
                plot_multi_subject_observation_curve(
                    ms_scaling, fig_dir, dataset=dataset, split=split
                )
            )
        vs_scaling = payload.get("vocabulary_scaling") or {}
        if len(vs_scaling) >= 2:
            generated.update(
                plot_vocabulary_scaling_curve(
                    vs_scaling, fig_dir, dataset=dataset, split=split
                )
            )
        return generated
    if status == "evaluation_completed" or "metrics" in payload:
        return plot_evaluation_figures(payload, fig_dir)

    raise ValueError(f"无法识别的评估 JSON 类型：{target}")
