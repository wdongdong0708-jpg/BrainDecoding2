"""Plot the matched Pallier2025 validation context-radius experiment."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path

from braindecoding.reporting import study_report_dir

RADIUS_VALUES = (0, 1, 2, 4, 8)
QUERY_COUNT = 16960
VOCABULARY_SIZE = "50"
METRIC = "macro_top10"
REPORT_DIR = study_report_dir("pallier2025", "local_window")


def evaluation_path(root: Path, radius: int | None, seed: int = 0) -> Path:
    if radius is None:
        return (
            root
            / "outputs/word_decoding/pallier2025/sub01-10/"
            f"main_context_warmstart/seed-{seed:03d}/evaluation/val.json"
        )
    return (
        root
        / "outputs/word_decoding/pallier2025/local_window/sub01-10/"
        f"local_window_r{radius}/seed-{seed:03d}/evaluation/val.json"
    )


def load_curve(root: Path, seed: int = 0) -> list[dict]:
    records = []
    reference = None
    warm_start_sha = None
    for radius in (*RADIUS_VALUES, None):
        path = evaluation_path(root, radius, seed)
        if not path.is_file():
            raise FileNotFoundError(f"缺少完整验证结果：{path}")
        result = json.loads(path.read_text(encoding="utf-8"))
        summary_path = path.parents[1] / "training_summary.json"
        if not summary_path.is_file():
            raise FileNotFoundError(f"缺少完整训练摘要：{summary_path}")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        expected_id = (
            "main_context_warmstart"
            if radius is None
            else f"local_window_r{radius}"
        )
        experiment = result["experiment"]
        data = result["data"]
        vocabulary = result["vocabularies"][VOCABULARY_SIZE]
        if (
            result["status"] != "evaluation_completed"
            or experiment["dataset"] != "pallier2025"
            or experiment["experiment_id"] != expected_id
            or experiment["seed"] != seed
            or data["split"] != "val"
            or data["query_count"] != QUERY_COUNT
            or vocabulary["candidate_count"] != 50
            or summary["status"] != "training_completed"
            or summary["experiment"] != experiment
            or summary["checkpoint"]["best"]["sha256"]
            != result["checkpoint"]["sha256"]
            or summary["selection"]["metric"]
            != "retrieval_acc10_vocab=pallier2025_50_macro"
            or summary["training"]["budget"] != {"type": "epochs", "value": 30}
            or summary["details"]["train_rows"] != 115290
            or summary["details"]["validation_rows"] != QUERY_COUNT
        ):
            raise ValueError(f"Pallier 局部窗口验证合同不一致：{path}")
        fingerprint = (
            data["event_table_sha256"],
            tuple(data["subjects"]),
            vocabulary["manifest_sha256"],
        )
        retrieval_queries = int(summary["details"]["best_validation"]["query_count"])
        if not 0 < retrieval_queries <= QUERY_COUNT:
            raise ValueError(f"冻结词表的有效查询数无效：{path}")
        fingerprint = (*fingerprint, retrieval_queries)
        if reference is None:
            reference = fingerprint
        elif fingerprint != reference:
            raise ValueError(f"验证事件、受试者或候选词表不一致：{path}")
        source_sha = summary["details"]["initialization_audit"][
            "source_checkpoint_sha256"
        ]
        if warm_start_sha is None:
            warm_start_sha = source_sha
        elif source_sha != warm_start_sha:
            raise ValueError(f"CNN 预热检查点不一致：{path}")
        score = vocabulary["retrieval"][METRIC]
        if score is None or not 0.0 <= float(score) <= 1.0:
            raise ValueError(f"缺少有效的 {METRIC}：{path}")
        records.append(
            {
                "radius": "full" if radius is None else radius,
                "window_words": "full" if radius is None else 2 * radius + 1,
                "validation_queries": data["query_count"],
                "retrieval_queries": retrieval_queries,
                "macro_top10_vocab50": float(score),
                "checkpoint_sha256": result["checkpoint"]["sha256"],
                "evaluation_path": str(path),
            }
        )
    return records


def load_multiseed_curve(root: Path, seeds=(0, 1, 2)) -> list[dict]:
    seeds = tuple(int(seed) for seed in seeds)
    if len(seeds) < 2 or len(seeds) != len(set(seeds)):
        raise ValueError("多种子曲线至少需要两个不同的随机种子。")
    curves = {seed: load_curve(root, seed=seed) for seed in seeds}
    reference = None
    for seed in seeds:
        path = evaluation_path(root, 0, seed)
        result = json.loads(path.read_text(encoding="utf-8"))
        summary = json.loads(
            (path.parents[1] / "training_summary.json").read_text(encoding="utf-8")
        )
        fingerprint = (
            result["data"]["event_table_sha256"],
            tuple(result["data"]["subjects"]),
            result["vocabularies"][VOCABULARY_SIZE]["manifest_sha256"],
            summary["details"]["initialization_audit"]["source_checkpoint_sha256"],
        )
        if reference is None:
            reference = fingerprint
        elif fingerprint != reference:
            raise ValueError("多种子实验的数据、词表或 CNN 预热检查点不一致。")
    rows = []
    for index, radius in enumerate((*RADIUS_VALUES, None)):
        points = [curves[seed][index] for seed in seeds]
        if len({point["retrieval_queries"] for point in points}) != 1:
            raise ValueError("多种子实验的冻结词表有效查询数不一致。")
        scores = [point["macro_top10_vocab50"] for point in points]
        rows.append(
            {
                "radius": "full" if radius is None else radius,
                "window_words": points[0]["window_words"],
                **{f"seed_{seed}": score for seed, score in zip(seeds, scores)},
                "mean_macro_top10_vocab50": statistics.mean(scores),
                "sd_macro_top10_vocab50": statistics.stdev(scores),
                "validation_queries_per_seed": QUERY_COUNT,
                "retrieval_queries_per_seed": points[0]["retrieval_queries"],
            }
        )
    return rows


def write_multiseed_curve(root: Path, output_dir: Path, seeds=(0, 1, 2)) -> tuple[Path, Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = load_multiseed_curve(root, seeds=seeds)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "multiseed_curve.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    local = rows[:-1]
    full = rows[-1]
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.errorbar(
        [row["radius"] for row in local],
        [100 * row["mean_macro_top10_vocab50"] for row in local],
        yerr=[100 * row["sd_macro_top10_vocab50"] for row in local],
        marker="o",
        capsize=3,
        linewidth=2,
        label="Local window: mean ± SD",
    )
    full_mean = 100 * full["mean_macro_top10_vocab50"]
    full_sd = 100 * full["sd_macro_top10_vocab50"]
    ax.axhline(full_mean, color="0.35", linestyle="--", label="Full context: mean")
    ax.axhspan(full_mean - full_sd, full_mean + full_sd, color="0.35", alpha=0.12)
    ax.set_xticks(RADIUS_VALUES)
    ax.set_xlabel("Context radius (words on each side)")
    ax.set_ylabel("Validation macro Top-10, 50-word vocabulary (%)")
    ax.set_title(f"Pallier2025: context size across {len(tuple(seeds))} seeds")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    figure_path = output_dir / "multiseed_curve.png"
    fig.savefig(figure_path, dpi=200)
    plt.close(fig)
    return csv_path, figure_path


def write_curve(root: Path, output_dir: Path, seed: int = 0) -> tuple[Path, Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    records = load_curve(root, seed=seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"seed-{seed:03d}_curve"
    csv_path = output_dir / f"{stem}.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)

    local = records[:-1]
    full = records[-1]
    local_scores = [100 * record["macro_top10_vocab50"] for record in local]
    full_score = 100 * full["macro_top10_vocab50"]
    best_local_index = max(range(len(local_scores)), key=local_scores.__getitem__)
    fig, ax = plt.subplots(figsize=(9.2, 5.1))
    ax.plot(
        range(len(local)),
        local_scores,
        color="#176B87",
        marker="o",
        markersize=7,
        linewidth=2.3,
        label="Local context",
    )
    ax.scatter(
        [len(local)],
        [full_score],
        color="#C44E36",
        marker="D",
        s=85,
        zorder=4,
        label="Full sentence",
    )
    ax.axhline(full_score, color="#C44E36", linestyle="--", alpha=0.55)
    ax.axvline(len(local) - 0.5, color="0.7", linestyle=":", linewidth=1)
    for index, score in enumerate((*local_scores, full_score)):
        ax.annotate(
            f"{score:.1f}%",
            (index, score),
            xytext=(0, 9),
            textcoords="offset points",
            ha="center",
            fontsize=10,
            fontweight="bold" if index == best_local_index else "normal",
        )
    labels = [
        f"±{record['radius']}\n({record['window_words']} word{'s' if record['window_words'] != 1 else ''})"
        for record in local
    ] + ["Full\nsentence"]
    ax.set_xticks(range(len(records)), labels)
    ax.set_xlim(-0.35, len(records) - 0.65)
    ax.set_ylim(30, 81)
    ax.set_ylabel("Macro Top-10 accuracy (%) · 50-word vocabulary")
    ax.set_title(
        f"Pallier2025 · seed {seed} · 50-word vocabulary "
        f"({records[0]['retrieval_queries']:,} queries)"
    )
    ax.grid(axis="y", alpha=0.22)
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    figure_path = output_dir / f"{stem}.png"
    fig.savefig(figure_path, dpi=200)
    plt.close(fig)
    report_path = output_dir / f"seed-{seed:03d}_comparison.md"
    lines = [
        f"# Pallier2025 seed {seed}：局部窗口与整句上下文对比",
        "",
        f"![验证集对比]({stem}.png)",
        "",
        "| 上下文 | 输入词数 | 验证集 macro Top-10（50 词表） | 相对整句 |",
        "|---|---:|---:|---:|",
    ]
    for record in records:
        context = "整句" if record["radius"] == "full" else f"±{record['radius']}"
        words = "整句" if record["window_words"] == "full" else str(record["window_words"])
        score = 100 * record["macro_top10_vocab50"]
        difference = score - full_score
        lines.append(f"| {context} | {words} | {score:.2f}% | {difference:+.2f} 个百分点 |")
    lines.extend(
        (
            "",
            "六组均使用同一 Pallier2025 validation 划分（16,960 个词事件）、"
            f"冻结 50 词表（{records[0]['retrieval_queries']:,} 个可评估查询）"
            "和同一个 CNN 预热检查点。每档局部窗口均按对应窗口单独训练。",
            "这里是单个上下文训练种子的结果；三个种子的均值和标准差将用于检验稳健性。",
            "",
        )
    )
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return csv_path, figure_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    args = parser.parse_args()
    root = args.root.resolve()
    output_dir = args.output_dir or REPORT_DIR
    if len(args.seeds) == 1:
        csv_path, figure_path = write_curve(root, output_dir, seed=args.seeds[0])
    else:
        csv_path, figure_path = write_multiseed_curve(root, output_dir, args.seeds)
    print(csv_path)
    print(figure_path)


if __name__ == "__main__":
    main()
