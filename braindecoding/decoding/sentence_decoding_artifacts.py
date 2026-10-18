"""Candidate diagnostics and completed-result figures from canonical v2 data."""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path

import numpy as np

from braindecoding.config import PROJECT_ROOT
from braindecoding.data.text import load_text_embedding_cache
from braindecoding.unified_sentence_v2 import OUT, encoded_path, manifest, read, save, sha256


def csv_write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def candidates(m):
    rows, source_rows = [], []
    for dataset, info in m["datasets"].items():
        embeddings = load_text_embedding_cache(info["embedding_cache"])
        bank = np.stack([embeddings[w] for w in info["vocabulary"]])
        index = {w: i for i, w in enumerate(info["vocabulary"])}
        for seed in m["settings"]["protocol"]["seeds"]:
            path = encoded_path(dataset, seed)
            archive = np.load(path)
            ids = list(map(str, archive["event_ids"]))
            lookup = {e: i for i, e in enumerate(ids)}
            source_rows.append(dict(path=path.relative_to(PROJECT_ROOT).as_posix(), sha256=sha256(path)))
            metadata = {}
            for group in info["groups"]:
                for pos, (event, word) in enumerate(zip(group["event_ids"], group["reference"])):
                    metadata[event] = dict(subject=group["subject"], partition=group["partition"],
                                           available=min(2, len(group["event_ids"])-1-pos), target=index.get(word, -1))
            targets = np.array([metadata[e]["target"] for e in ids])
            for horizon in range(3):
                predictions = archive[f"k{horizon}"]
                hits = np.zeros(len(ids), dtype=bool)
                for start in range(0, len(ids), 512):
                    scores = predictions[start:start+512].astype(np.float64) @ bank.astype(np.float64).T
                    top = np.argsort(-scores, axis=1, kind="stable")[:, :20]
                    hits[start:start+512] = (top == targets[start:start+512, None]).any(1)
                for partition in ("calibration", "evaluation", "all_val"):
                    by_subject = defaultdict(list)
                    for i, event in enumerate(ids):
                        meta = metadata[event]
                        if meta["available"] >= horizon and (partition == "all_val" or meta["partition"] == partition):
                            by_subject[meta["subject"]].append(i)
                    for subject, selected in sorted(by_subject.items()):
                        target = targets[selected]
                        covered = target >= 0
                        rows.append(dict(dataset=dataset, seed=seed, partition=partition, subject=subject,
                            horizon=horizon, candidate_count=len(index), shortlist=20, queries=len(selected),
                            oov_queries=int((~covered).sum()), hits=int(hits[selected].sum()),
                            recall20=float(hits[selected].mean()), oov_retained_as_errors=True))
    prefix = PROJECT_ROOT / "data/results/exports/unified_sentence_v2_"
    csv_write(Path(str(prefix)+"candidate_recall.csv"), rows)
    summary = []
    for dataset in m["datasets"]:
        for partition in ("calibration", "evaluation", "all_val"):
            for horizon in range(3):
                selected = [r for r in rows if r["dataset"] == dataset and r["partition"] == partition and r["horizon"] == horizon]
                values = [np.mean([r["recall20"] for r in selected if r["seed"] == seed])
                          for seed in m["settings"]["protocol"]["seeds"]]
                summary.append(dict(dataset=dataset, partition=partition, horizon=horizon,
                    subject_equal_recall20=float(np.mean(values)), seed_sd=float(np.std(values, ddof=1)),
                    queries_per_seed=sum(r["queries"] for r in selected if r["seed"] == 0),
                    oov_per_seed=sum(r["oov_queries"] for r in selected if r["seed"] == 0)))
    csv_write(Path(str(prefix)+"candidate_summary.csv"), summary)
    save(Path(str(prefix)+"candidate_sources.json"), dict(manifest_sha256=m["manifest_sha256"], sources=source_rows,
        implementation_sha256=sha256(Path(__file__)), test_evaluated=False, oov_retained_as_errors=True))
    return summary


def figures(m):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import rcParams

    report_path = PROJECT_ROOT / "data/results/studies/unified_sentence_v2/results.json"
    report = read(report_path)
    if report["status"] != "completed" or report["manifest_sha256"] != m["manifest_sha256"]:
        raise ValueError("Figures require complete matching experiment")
    with (PROJECT_ROOT / "data/results/exports/unified_sentence_v2_summary.csv").open(encoding="utf-8-sig") as f:
        summary = list(csv.DictReader(f))
    with (PROJECT_ROOT / "data/results/exports/unified_sentence_v2_statistics.csv").open(encoding="utf-8-sig") as f:
        stats = list(csv.DictReader(f))
    rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "pdf.fonttype": 42,
                     "svg.fonttype": "none", "axes.spines.top": False, "axes.spines.right": False})
    datasets = list(m["datasets"])
    names = {"pallier2025": "Pallier2025", "chineseeeg2_littleprince": "ChineseEEG-2"}
    colors = ["#287C8E", "#B86632"]
    destination = PROJECT_ROOT / "data/results/figures/unified_sentence_v2"
    destination.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(10.6, 3.6), gridspec_kw={"width_ratios": [1.4, 1.4, 1.]})
    for axis, dataset, color in zip(axes[:2], datasets, colors):
        selected = [r for r in summary if r["dataset"] == dataset and r["branch"] == "open"]
        values = [float(next(r for r in selected if r["alignment"] == "real" and r["condition"] == c)["sentence_bleu1_mean_mean"]) * 100 for c in "ABC"]
        shifted = [float(r["sentence_bleu1_mean_mean"]) * 100 for r in selected if r["alignment"].startswith("shift-") and r["condition"] == "C"]
        pure = float(next(r for r in selected if r["alignment"] == "pure_language")["sentence_bleu1_mean_mean"]) * 100
        values.extend([float(np.mean(shifted)), pure])
        axis.bar(range(5), values, color=[color]*3 + ["#909090", "#BBBBBB"], width=.65)
        axis.set_xticks(range(5), ["A", "B", "C", "Shifted\nC", "LM only"])
        axis.set_title(names[dataset])
        axis.set_ylabel("Mean sentence BLEU-1 (%)")
        axis.set_ylim(0, max(values) * 1.22 if max(values) else 1)
        for x, value in enumerate(values):
            axis.text(x, value + max(values) * .025, f"{value:.1f}", ha="center", fontsize=8)
    for i, (dataset, color) in enumerate(zip(datasets, colors)):
        row = next(r for r in stats if r["dataset"] == dataset and r["comparison"] == "C_minus_B")
        effect, low, high = [float(row[k])*100 for k in ("effect", "ci_lower", "ci_upper")]
        axes[2].hlines(i, low, high, color=color, linewidth=1.3)
        axes[2].vlines([low, high], i-.06, i+.06, color=color, linewidth=1.)
        axes[2].plot(effect, i, "o", color=color)
        axes[2].text(high + .15, i, f"p={float(row['p_holm']):.3g}", va="center", fontsize=8)
    axes[2].set_yticks(range(len(datasets)), [names[d] for d in datasets])
    axes[2].axvline(0, color="#777777", linewidth=.8, linestyle="--")
    axes[2].set_xlabel("C − B (percentage points)\n95% material-block bootstrap CI")
    axes[2].set_title("Neural refresh increment")
    axes[2].set_ylim(-.6, len(datasets)-.4)
    fig.tight_layout()
    for suffix in ("pdf", "svg", "png"):
        fig.savefig(destination / f"sentence_results.{suffix}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    save(destination / "sources.json", dict(manifest_sha256=m["manifest_sha256"],
        report_sha256=sha256(report_path), implementation_sha256=sha256(Path(__file__)),
        figure_files=[dict(path=p.relative_to(PROJECT_ROOT).as_posix(), sha256=sha256(p))
                      for p in destination.glob("sentence_results.*")]))
    examples(m)


def examples(m):
    selected = {}
    for dataset, info in m["datasets"].items():
        decoded, sources = {}, []
        root = OUT / dataset
        for path in sorted((root / "seed-000/open/real").glob("chunk-*.json")):
            chunk = read(path)
            if chunk["manifest_sha256"] != m["manifest_sha256"]:
                raise ValueError("Example source does not match the completed experiment")
            sources.append(dict(path=path.relative_to(PROJECT_ROOT).as_posix(), sha256=sha256(path)))
            decoded.update({r["group"]: r["result"] for r in chunk["records"]})
        records = []
        for group in info["groups"]:
            if group["partition"] != "evaluation":
                continue
            result = decoded[group["group"]]
            reference = group["reference"]
            initial = [result["trace"][i]["C"][i] for i in range(len(reference))]
            final = result["final"]["C"]
            repaired = [i for i, (truth, before, after) in enumerate(zip(reference, initial, final))
                        if before != truth and after == truth]
            harmed = [i for i, (truth, before, after) in enumerate(zip(reference, initial, final))
                      if before == truth and after != truth]
            records.append(dict(group=group["group"], subject=group["subject"], block=group["block"],
                reference=reference, initial_C=initial, final=result["final"],
                repaired_positions_zero_based=repaired, harmed_positions_zero_based=harmed,
                trace=result["trace"]))
        choices = [("first_evaluation_group", records[0]),
                   ("most_C_repairs", sorted(records, key=lambda r: (-len(r["repaired_positions_zero_based"]), len(r["harmed_positions_zero_based"]), r["group"]))[0]),
                   ("most_C_harms", sorted(records, key=lambda r: (-len(r["harmed_positions_zero_based"]), len(r["repaired_positions_zero_based"]), r["group"]))[0])]
        selected[dataset] = dict(seed=0, branch="open", weight=read(root / "calibration.json")["selected_weight"],
            selection_is_for_illustration_not_performance_estimation=True,
            examples=[dict(selection=label, **record) for label, record in choices], sources=sources)
    save(PROJECT_ROOT / "data/results/studies/unified_sentence_v2/examples.json",
         dict(manifest_sha256=m["manifest_sha256"], datasets=selected, test_evaluated=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", action="store_true")
    parser.add_argument("--figures", action="store_true")
    args = parser.parse_args()
    m = manifest()
    if args.candidates:
        print(json.dumps(candidates(m), ensure_ascii=False), flush=True)
    if args.figures:
        figures(m)


if __name__ == "__main__":
    main()
