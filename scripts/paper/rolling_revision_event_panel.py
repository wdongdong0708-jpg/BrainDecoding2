"""Draw a rolling-performance curve above paired word-event transition bars."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from statistics import mean, stdev

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from braindecoding.config import PROJECT_ROOT
from scripts.paper.paper_style import CURVE_STYLES, configure_matplotlib, save_figure


DEFAULT_DATA = PROJECT_ROOT / "data/results" / "exports" / "rolling_revision_transitions.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "paper" / "figures" / "pallier2025_rolling_revision_events"
CORRECTED_COLOR = "#168C8C"
REGRESSED_COLOR = "#D97706"


def load_data(path: Path) -> tuple[dict, dict[int, dict[int, float]], dict[tuple[int, int], dict[int, dict]]]:
    """Validate paired counts and return per-seed curves and step events."""
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError("No transition rows in exported data.")
    identity = {(row["dataset"], row["split"], row["vocabulary_size"],
                 row["top_k"], row["target_set"]) for row in rows}
    if len(identity) != 1:
        raise ValueError("Figure data mixes evaluation contracts.")
    dataset, split, size, top_k, target_set = next(iter(identity))
    if split not in {"val", "test"} or int(size) <= 0 or not 0 < int(top_k) <= int(size) or not target_set:
        raise ValueError("Invalid paired evaluation contract.")
    curves: dict[int, dict[int, float]] = {}
    steps: dict[tuple[int, int], dict[int, dict]] = {}
    eligibility = set()
    for row in rows:
        seed = int(row["seed"])
        before, after = int(row["from_delay"]), int(row["to_delay"])
        if (before, after) not in {(0, 1), (1, 2)}:
            raise ValueError(f"Unexpected revision transition: K{before} to K{after}.")
        counts = {key: int(row[key]) for key in (
            "eligible_word_events", "wrong_to_right", "right_to_wrong",
            "right_to_right", "wrong_to_wrong",
        )}
        n = counts["eligible_word_events"]
        if n <= 0 or sum(counts[key] for key in counts if key != "eligible_word_events") != n:
            raise ValueError("Paired event counts do not partition the eligible cohort.")
        if not np.isclose((counts["right_to_right"] + counts["right_to_wrong"]) / n,
                          float(row["micro_before"]), atol=1e-8, rtol=0):
            raise ValueError("Before-hit count differs from micro Top-10.")
        if not np.isclose((counts["right_to_right"] + counts["wrong_to_right"]) / n,
                          float(row["micro_after"]), atol=1e-8, rtol=0):
            raise ValueError("After-hit count differs from micro Top-10.")
        points = curves.setdefault(seed, {})
        for delay, value in ((before, float(row["macro_before"])),
                             (after, float(row["macro_after"]))):
            if delay in points and not np.isclose(points[delay], value, atol=1e-10, rtol=0):
                raise ValueError("Adjacent steps disagree on their shared macro score.")
            points[delay] = value
        step = steps.setdefault((before, after), {})
        if seed in step:
            raise ValueError("Duplicate seed/transition row.")
        step[seed] = counts
        eligibility.add(n)
    seed_set = set(curves)
    if not seed_set or any(set(points) != {0, 1, 2} for points in curves.values()):
        raise ValueError("Each seed needs K0, K1, and K2 performance.")
    if any(set(step) != seed_set for step in steps.values()) or set(steps) != {(0, 1), (1, 2)}:
        raise ValueError("Each seed needs both paired transitions.")
    if len(eligibility) != 1:
        raise ValueError("Eligible event count must be shared across seeds and steps.")
    return {"dataset": dataset, "split": split, "vocabulary_size": int(size),
            "top_k": int(top_k), "target_set": target_set}, curves, steps


def build_figure(metadata: dict, curves: dict[int, dict[int, float]],
                 steps: dict[tuple[int, int], dict[int, dict]]):
    """Use a common seed cohort for the upper macro curve and lower event bars."""
    with mpl.rc_context():
        configure_matplotlib("reference")
        figure, (curve_ax, event_ax) = plt.subplots(
            2, 1, figsize=(3.5, 4.25),
            gridspec_kw={"height_ratios": [1.0, 1.05], "hspace": 0.45},
        )
        seeds = sorted(curves)
        x = np.arange(3)
        color = CURVE_STYLES["rolling"]["color"]
        for seed in seeds:
            y = [curves[seed][delay] * 100 for delay in x]
            curve_ax.plot(x, y, color=color, alpha=0.27, linewidth=0.8,
                          marker="o", markersize=2.5, zorder=2)
        means = [mean(curves[seed][delay] for seed in seeds) * 100 for delay in x]
        curve_ax.plot(x, means, color=color, linewidth=1.9, marker="o", markersize=4.5,
                      markerfacecolor="white", markeredgewidth=1.15, zorder=3)
        curve_ax.set_xlim(-0.14, 2.14)
        curve_ax.set_xticks(x, [f"K{delay}" for delay in x])
        curve_ax.set_ylabel(f"Macro Top-{metadata['top_k']} (%)")
        curve_ax.set_title(f"{metadata['dataset']} · rolling revision", pad=7)
        curve_ax.text(0.98, 0.04, "thin: seeds   bold: mean", ha="right", va="bottom",
                      transform=curve_ax.transAxes, fontsize=6.6, color="#555555")
        curve_ax.grid(axis="x", visible=False)
        curve_ax.margins(y=0.16)

        centers = np.arange(2)
        width = 0.32
        highest_bar = 0.0
        for offset, key, label, bar_color in (
            (-width / 2, "wrong_to_right", "Wrong → right", CORRECTED_COLOR),
            (width / 2, "right_to_wrong", "Right → wrong", REGRESSED_COLOR),
        ):
            values = [mean(steps[step][seed][key] for seed in seeds)
                      for step in ((0, 1), (1, 2))]
            errors = [stdev(steps[step][seed][key] for seed in seeds) if len(seeds) > 1 else 0
                      for step in ((0, 1), (1, 2))]
            highest_bar = max(highest_bar, *(value + error for value, error in zip(values, errors)))
            bars = event_ax.bar(centers + offset, values, width=width, label=label,
                                color=bar_color, alpha=0.86, edgecolor="white", linewidth=0.4,
                                yerr=errors, capsize=2.2, error_kw={"lw": 0.7, "capthick": 0.7})
            for bar, value, error in zip(bars, values, errors):
                event_ax.text(bar.get_x() + bar.get_width() / 2, value + error,
                              f"{value:,.0f}", ha="center", va="bottom", fontsize=6.5)
        event_ax.set_xticks(centers, ["K0 → K1", "K1 → K2"])
        event_ax.set_xlim(-0.58, 1.58)
        event_ax.set_ylabel("Word events per seed")
        event_ax.set_ylim(0, highest_bar * 1.22)
        event_ax.grid(axis="x", visible=False)
        event_ax.legend(loc="upper right", ncol=2, columnspacing=0.8, handlelength=1.1,
                        handletextpad=0.4, bbox_to_anchor=(1, 1.13))
        n = steps[(0, 1)][seeds[0]]["eligible_word_events"]
        target_label = ("K2-eligible" if metadata["target_set"] == "common_delay2"
                        else metadata["target_set"].replace("_", " "))
        figure.text(0.5, 0.015,
                    f"Paired {target_label} Top-{metadata['top_k']} events (N = {n:,}/seed); "
                    f"bars: mean ± SD, n = {len(seeds)} seeds",
                    ha="center", va="bottom", fontsize=6.6, color="#555555")
        figure.subplots_adjust(left=0.18, right=0.98, top=0.92, bottom=0.12)
        return figure


def main(argv: list[str] | None = None) -> tuple[Path, ...]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    if not args.data.resolve().is_relative_to((PROJECT_ROOT / "data/results" / "exports").resolve()):
        parser.error("Paper plots must consume reports/exports CSVs.")
    metadata, curves, steps = load_data(args.data)
    figure = build_figure(metadata, curves, steps)
    try:
        return save_figure(figure, args.output, formats=("pdf", "svg", "png"))
    finally:
        plt.close(figure)


if __name__ == "__main__":
    for path in main():
        print(path)
