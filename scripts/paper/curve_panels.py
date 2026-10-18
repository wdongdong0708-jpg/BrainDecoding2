"""Render reusable curve panels from an export CSV and a presentation config.

No experiment aggregation takes place here. Each selected row is one exported
point (or one horizontal reference); duplicate points are rejected.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

try:
    from .paper_style import CURVE_STYLES, configure_matplotlib, save_figure
except ImportError:
    from paper_style import CURVE_STYLES, configure_matplotlib, save_figure

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPORT_DIR = PROJECT_ROOT / "data/results" / "exports"


def _matches(row: dict, filters: dict) -> bool:
    return all(row[key] == str(value) for key, value in filters.items())


def load_data(path: str | Path, config: dict, split: str) -> list[dict]:
    """Read only the requested split; never fall back or average duplicate rows."""
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        columns = config["columns"]
        required = {columns[key] for key in ("panel", "x", "y")}
        required.add(columns.get("split", "split"))
        required.update(config.get("filters", {}))
        for series in config["series"]:
            required.update(series.get("filters", {}))
            if series.get("background_runs"):
                required.update(columns[key] for key in ("seeds", "seed_values", "n"))
        if config.get("uncertainty", "none") != "none":
            required.update(columns[key] for key in ("sd", "n"))
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"Missing CSV columns: {sorted(missing)}")
        rows = [row for row in reader if row[columns.get("split", "split")] == split]
    rows = [row for row in rows if _matches(row, config.get("filters", {}))]
    if not rows:
        raise ValueError(f"No exported data for split={split!r} and requested filters; no fallback.")
    return rows


def _number(value, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite.")
    return result


def _point(row: dict, config: dict) -> tuple[float, float, int | None]:
    columns = config["columns"]
    scale = _number(config.get("scale", 1), "scale")
    if scale <= 0:
        raise ValueError("scale must be positive.")
    value = _number(row[columns["y"]], "y") * scale
    if config.get("uncertainty", "none") == "none":
        return value, math.nan, None
    n_value = _number(row[columns["n"]], "n")
    if not n_value.is_integer() or n_value < 1:
        raise ValueError("n must be a positive integer.")
    n = int(n_value)
    sd_text = row[columns["sd"]]
    if n == 1:
        if sd_text != "":
            raise ValueError("A single run must have blank SD, not a fabricated zero.")
        return value, math.nan, n
    sd = _number(sd_text, "SD") * scale
    if sd < 0:
        raise ValueError("SD must be nonnegative.")
    return value, sd, n


def _annotate_line(axis, x: list[float], values: list[float], spec: dict) -> None:
    """Label measured means and adjacent changes, without inferring significance."""
    annotation = spec.get("annotations")
    if not annotation:
        return
    if annotation.get("point_values"):
        offsets = annotation.get("point_offsets", [[0, 9]] * len(values))
        if len(offsets) != len(values):
            raise ValueError("point_offsets must match x_values.")
        for position, value, offset in zip(x, values, offsets):
            if not math.isfinite(value):
                continue
            axis.annotate(
                f"{value:.2f}", (position, value), xytext=offset,
                textcoords="offset points", ha="center", va="center",
                fontsize=7, color="#273443", zorder=6,
                bbox={"boxstyle": "round,pad=0.12", "facecolor": "white",
                      "edgecolor": "none", "alpha": 0.9},
            )
    if annotation.get("step_gains"):
        offsets = annotation.get("step_offsets", [[0, 9]] * (len(values) - 1))
        if len(offsets) != len(values) - 1:
            raise ValueError("step_offsets must match adjacent x pairs.")
        for index, offset in enumerate(offsets):
            left, right = values[index:index + 2]
            if not (math.isfinite(left) and math.isfinite(right)):
                continue
            middle = ((x[index] + x[index + 1]) / 2, (left + right) / 2)
            axis.annotate(
                f"{right - left:+.2f} pp", middle, xytext=offset,
                textcoords="offset points", ha="center", va="center",
                fontsize=7.5, fontweight="bold", color="#2C5F9C", zorder=6,
                bbox={"boxstyle": "round,pad=0.13", "facecolor": "white",
                      "edgecolor": "none", "alpha": 0.92},
            )


def _plot_background_runs(axis, x: list[float], x_keys: list[str], rows: list[dict],
                          columns: dict, color: str, scale: float) -> None:
    """Show measured seed trajectories behind the aggregate, with no jitter."""
    per_seed: dict[str, dict[str, float]] = {}
    expected_seeds: set[str] | None = None
    for row in rows:
        key = row[columns["x"]]
        if key not in x_keys:
            raise ValueError(f"Unexpected x={key!r} in individual-run data.")
        seed_ids = row[columns["seeds"]].split(";")
        seed_texts = row[columns["seed_values"]].split(";")
        if len(seed_ids) != len(seed_texts) or len(set(seed_ids)) != len(seed_ids):
            raise ValueError(f"Seed identities and values mismatch at x={key}.")
        if expected_seeds is None:
            expected_seeds = set(seed_ids)
        elif set(seed_ids) != expected_seeds:
            raise ValueError(f"Seed identities change across x at x={key}.")
        if len(seed_ids) != int(row[columns["n"]]):
            raise ValueError(f"Seed count mismatch at x={key}.")
        values = [_number(value, "seed value") * scale for value in seed_texts]
        aggregate = _number(row[columns["y"]], "mean") * scale
        if not math.isclose(sum(values) / len(values), aggregate, abs_tol=1e-8):
            raise ValueError(f"Seed mean does not match aggregate at x={key}.")
        for seed, value in zip(seed_ids, values):
            if key in per_seed.setdefault(seed, {}):
                raise ValueError(f"Duplicate seed {seed} at x={key}.")
            per_seed[seed][key] = value
    for seed in sorted(per_seed):
        axis.plot(x, [per_seed[seed].get(key, math.nan) for key in x_keys],
                  color=color, alpha=0.42, linewidth=0.8, marker="o",
                  markersize=2.5, markeredgewidth=0, zorder=1.5)


def build_figure(data: list[dict], config: dict):
    """Build 1..N panels with a shared legend, explicit gaps, and optional SD."""
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    columns = config["columns"]
    panels = config.get("panels") or list(dict.fromkeys(row[columns["panel"]] for row in data))
    if not panels or len(set(panels)) != len(panels):
        raise ValueError("panels must be nonempty and unique.")
    x_values = config["x_values"]
    if not x_values or len(set(map(str, x_values))) != len(x_values):
        raise ValueError("x_values must be nonempty and unique.")
    x_keys = list(map(str, x_values))
    spacing = config.get("x_spacing", "numeric")
    if spacing not in {"numeric", "categorical"}:
        raise ValueError("x_spacing must be numeric or categorical.")
    x = list(range(len(x_values))) if spacing == "categorical" else [_number(v, "x") for v in x_values]
    if any(right <= left for left, right in zip(x, x[1:])):
        raise ValueError("Numeric x_values must be strictly increasing.")
    uncertainty = config.get("uncertainty", "none")
    if uncertainty not in {"none", "sd_bars", "sd_band"}:
        raise ValueError("uncertainty must be none, sd_bars, or sd_band.")
    series_specs = config["series"]
    if not series_specs or len({s["label"] for s in series_specs}) != len(series_specs):
        raise ValueError("Series labels must be nonempty and unique.")
    ncols = min(int(config.get("max_columns", 3)), len(panels))
    if ncols < 1:
        raise ValueError("max_columns must be positive.")
    nrows = math.ceil(len(panels) / ncols)
    width = 3.5 if ncols == 1 else 7.2
    handles, counts = {}, set()
    legend_columns = min(len(series_specs), 3)
    legend_rows = math.ceil(len(series_specs) / legend_columns)
    legend_height = 0.25 + 0.17 * (legend_rows - 1)
    with mpl.rc_context():
        configure_matplotlib(config.get("profile", "reference"))
        figure = plt.figure(figsize=(width, 2.1 * nrows + 0.65 + 0.23 * (legend_rows - 1)),
                            layout="constrained")
        try:
            grid = figure.add_gridspec(nrows + 1, ncols,
                                      height_ratios=[1] * nrows + [legend_height])
            first_axis = None
            for index, panel in enumerate(panels):
                axis = figure.add_subplot(
                    grid[index // ncols, index % ncols],
                    sharey=first_axis if config.get("share_y", True) else None,
                )
                if first_axis is None:
                    first_axis = axis
                panel_rows = [row for row in data if row[columns["panel"]] == panel]
                if not panel_rows:
                    raise ValueError(f"No exported rows for requested panel {panel!r}.")
                for spec in series_specs:
                    rows = [row for row in panel_rows if _matches(row, spec.get("filters", {}))]
                    if not rows:
                        raise ValueError(f"Missing series {spec['label']!r} in {panel!r}.")
                    style = dict(CURVE_STYLES[spec["style"]])
                    label = spec["label"]
                    kind = spec.get("kind", "line")
                    if kind == "reference":
                        if len(rows) != 1:
                            raise ValueError(f"Reference {label!r} needs exactly one row per panel.")
                        value, sd, n = _point(rows[0], config)
                        counts.add(n)
                        style.pop("marker")
                        line = axis.axhline(value, **style, label=label, zorder=2)
                        if math.isfinite(sd):
                            axis.axhspan(value - sd, value + sd, color=style["color"], alpha=0.08, linewidth=0)
                    elif kind == "line":
                        by_x = {}
                        for row in rows:
                            key = row[columns["x"]]
                            if key not in x_keys:
                                raise ValueError(f"Unexpected x={key!r} for {label!r}; update x_values or filters.")
                            if key in by_x:
                                raise ValueError(f"Duplicate point for {panel}/{label}/x={key}; filter the evaluation contract.")
                            by_x[key] = _point(row, config)
                            counts.add(by_x[key][2])
                        values = [by_x.get(key, (math.nan, math.nan, None))[0] for key in x_keys]
                        errors = [by_x.get(key, (math.nan, math.nan, None))[1] for key in x_keys]
                        if spec.get("background_runs"):
                            _plot_background_runs(axis, x, x_keys, rows, columns,
                                                  style["color"], _number(config.get("scale", 1), "scale"))
                            style["linewidth"] = 1.8
                            style["markersize"] = 4.4
                        line, = axis.plot(x, values, **style, markerfacecolor="white", label=label, zorder=3)
                        if uncertainty == "sd_bars":
                            indices = [i for i, sd in enumerate(errors) if math.isfinite(sd)]
                            if indices:
                                axis.errorbar([x[i] for i in indices], [values[i] for i in indices],
                                              yerr=[errors[i] for i in indices], fmt="none",
                                              color=style["color"], capsize=2.2, linewidth=0.8, zorder=2)
                        elif uncertainty == "sd_band":
                            axis.fill_between(x, [v - sd for v, sd in zip(values, errors)],
                                              [v + sd for v, sd in zip(values, errors)],
                                              color=style["color"], alpha=0.13, linewidth=0)
                        _annotate_line(axis, x, values, spec)
                    else:
                        raise ValueError(f"Unknown series kind: {kind!r}")
                    handles.setdefault(label, line)
                panel_label = config.get("panel_labels", {}).get(panel, panel)
                title = panel_label if len(panels) == 1 else f"({chr(97 + index)}) {panel_label}"
                axis.set_title(title, pad=6)
                axis.set_xlabel(config["xlabel"])
                axis.set_xticks(x, config.get("x_labels", list(map(str, x_values))))
                axis.margins(x=0.07)
                if "ylim" in config:
                    axis.set_ylim(*config["ylim"])
                if index % ncols == 0 or not config.get("share_y", True):
                    axis.set_ylabel(config["ylabel"])
                elif config.get("share_y", True):
                    axis.tick_params(labelleft=False)
            legend_axis = figure.add_subplot(grid[-1, :])
            legend_axis.set_axis_off()
            legend_axis.legend(handles.values(), handles.keys(), loc="upper center",
                               ncol=legend_columns, handlelength=2.0,
                               columnspacing=0.9, handletextpad=0.4)
            if uncertainty != "none":
                ns = ", ".join(str(n) for n in sorted(n for n in counts if n is not None))
                note = f"Mean ± 1 SD across {config.get('replicate_unit', 'seeds')} (n = {ns})"
                if 1 in counts:
                    note += "; n = 1: point only, SD unavailable"
                legend_axis.text(0.5, -0.25, note, ha="center", va="top", fontsize=7,
                                 transform=legend_axis.transAxes)
            elif config.get("note"):
                legend_axis.text(0.5, -0.25, config["note"], ha="center", va="top",
                                 fontsize=7, transform=legend_axis.transAxes)
            return figure
        except Exception:
            plt.close(figure)
            raise


def main(argv: list[str] | None = None) -> tuple[Path, ...]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--data", type=Path, help="CSV in reports/exports; overrides config data.")
    parser.add_argument("--output", type=Path, help="Output stem; defaults to paper/figures/<config>.")
    args = parser.parse_args(argv)
    import matplotlib.pyplot as plt

    try:
        config = json.loads(args.config.read_text(encoding="utf-8"))
        data_path = (args.data or (PROJECT_ROOT / config["data"])).resolve()
        if not data_path.is_relative_to(EXPORT_DIR.resolve()):
            raise ValueError("Paper plots must consume CSVs in reports/exports/.")
        data = load_data(data_path, config, args.split)
        figure = build_figure(data, config)
        try:
            output = args.output or PROJECT_ROOT / "paper" / "figures" / args.config.stem
            return save_figure(figure, output, formats=("pdf", "svg", "png"))
        finally:
            plt.close(figure)
    except (ValueError, KeyError, OSError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    for path in main():
        print(path)
