"""Numerical/structural checks for the reusable renderer (test fixtures only)."""

import copy
import csv
import os

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pytest

from scripts.paper import curve_panels
from scripts.paper.paper_style import save_figure


@pytest.fixture
def curve_data():
    config = {
        "columns": {"panel": "dataset", "x": "x", "y": "mean", "sd": "sd", "n": "n"},
        "x_values": [0, 1, 2], "xlabel": "Delay (words)", "ylabel": "Score (%)",
        "scale": 100, "uncertainty": "sd_bars",
        "series": [
            {"label": "Rolling", "style": "rolling", "filters": {"method": "rolling"}},
            {"label": "Full", "style": "full", "kind": "reference", "filters": {"method": "full"}},
        ],
    }
    rows = [
        {"dataset": "Test panel", "split": "val", "method": method,
         "x": x, "mean": value, "sd": "0.02", "n": "3"}
        for method, x, value in [
            ("rolling", "0", "0.4"), ("rolling", "1", "0.5"),
            ("rolling", "2", "0.6"), ("full", "", "0.55"),
        ]
    ]
    return config, rows


def test_scaled_sd_references_gaps_and_local_style(curve_data, tmp_path):
    config, rows = curve_data
    assert curve_panels._point(rows[0], config) == (40, 2, 3)
    prior_style = dict(mpl.rcParams)
    rows.pop(1)
    fig = curve_panels.build_figure(rows, config)
    try:
        assert dict(mpl.rcParams) == prior_style
        axis = fig.axes[0]
        assert np.allclose(axis.lines[0].get_ydata(), [40, np.nan, 60], equal_nan=True)
        reference = next(line for line in axis.lines if line.get_label() == "Full")
        assert list(reference.get_ydata()) == pytest.approx([55, 55])
        assert axis.get_title() == "Test panel"
        paths = save_figure(fig, tmp_path / "test_render", ("pdf", "svg", "png"))
        assert all(path.stat().st_size > 0 for path in paths)
    finally:
        plt.close(fig)


def test_duplicate_and_missing_series_fail_without_leaking_figures(curve_data):
    config, rows = curve_data
    before = plt.get_fignums()
    with pytest.raises(ValueError, match="Duplicate"):
        curve_panels.build_figure(rows + [rows[0]], config)
    with pytest.raises(ValueError, match="Missing series"):
        curve_panels.build_figure(rows[:-1], config)
    config["panels"] = ["Not available"]
    with pytest.raises(ValueError, match="No exported rows"):
        curve_panels.build_figure(rows, config)
    assert plt.get_fignums() == before


def test_single_run_never_invents_uncertainty(curve_data):
    config, rows = curve_data
    row = {**rows[0], "n": "1", "sd": ""}
    assert np.isnan(curve_panels._point(row, config)[1])
    with pytest.raises(ValueError, match="single run"):
        curve_panels._point({**row, "sd": "0"}, config)
    with pytest.raises(ValueError):
        curve_panels._point({**row, "n": "3"}, config)
    with pytest.raises(ValueError, match="nonnegative"):
        curve_panels._point({**row, "n": "3", "sd": "-0.1"}, config)


def test_csv_filters_and_no_split_fallback(curve_data, tmp_path):
    config, rows = curve_data
    path = tmp_path / "exports.csv"
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    assert curve_panels.load_data(path, config, "val") == rows
    with pytest.raises(ValueError, match="no fallback"):
        curve_panels.load_data(path, config, "test")
    config["filters"] = {"missing_contract": "value"}
    with pytest.raises(ValueError, match="Missing CSV columns"):
        curve_panels.load_data(path, config, "val")


def test_five_panels_band_and_legend(curve_data):
    config, rows = curve_data
    config = copy.deepcopy(config)
    config["uncertainty"] = "sd_band"
    data = [{**row, "dataset": f"Panel {index}"} for index in range(5) for row in rows]
    fig = curve_panels.build_figure(data, config)
    try:
        assert len(fig.axes) == 6  # five data panels plus the shared legend
        assert len(fig.axes[-1].get_legend().get_texts()) == 2
        assert len(fig.axes[0].collections) == 1  # the line's SD band
    finally:
        plt.close(fig)


def test_annotations_use_exported_point_values_and_adjacent_differences(curve_data):
    config, rows = curve_data
    config = copy.deepcopy(config)
    config["series"][0]["annotations"] = {
        "point_values": True, "step_gains": True,
        "point_offsets": [[0, 8]] * 3,
        "step_offsets": [[0, 8]] * 2,
    }
    fig = curve_panels.build_figure(rows, config)
    try:
        labels = [annotation.get_text() for annotation in fig.axes[0].texts]
        assert labels == ["40.00", "50.00", "60.00", "+10.00 pp", "+10.00 pp"]
    finally:
        plt.close(fig)


def test_background_trajectories_use_reported_seed_values(curve_data):
    config, rows = curve_data
    config = copy.deepcopy(config)
    config["columns"].update({"seeds": "seeds", "seed_values": "seed_values"})
    config["uncertainty"] = "none"
    config["series"][0]["background_runs"] = True
    values = {
        "0": "0.39;0.40;0.41", "1": "0.49;0.50;0.51",
        "2": "0.59;0.60;0.61",
    }
    rows = [{**row, "seeds": "0;1;2", "seed_values": values.get(row["x"], "")}
            for row in rows]
    fig = curve_panels.build_figure(rows, config)
    try:
        axis = fig.axes[0]
        assert len(axis.lines) == 5  # three seeds, aggregate, reference
        assert list(axis.lines[0].get_ydata()) == pytest.approx([39, 49, 59])
        assert list(axis.lines[3].get_ydata()) == pytest.approx([40, 50, 60])
    finally:
        plt.close(fig)
    rows[0]["seed_values"] = "0.1;0.2;0.3"
    with pytest.raises(ValueError, match="Seed mean does not match"):
        curve_panels.build_figure(rows, config)
