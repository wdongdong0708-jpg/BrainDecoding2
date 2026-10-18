"""Shared rendering contract for the standalone publication figure suite."""
from __future__ import annotations

from contextlib import contextmanager
from hashlib import sha256
import json
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.font_manager import FontProperties

from braindecoding.config import PROJECT_ROOT

OUT = PROJECT_ROOT / "paper" / "figures" / "publication_suite"
EXPORT = PROJECT_ROOT / "data/results" / "exports"
INK, MUTED, RULE = "#26343D", "#586873", "#D5DDE2"
BLUE, TEAL, ORANGE, GRAY, PURPLE = "#356F99", "#287F82", "#B86A50", "#97A4AE", "#86699B"
DATA_COLORS = {"pallier2025": BLUE, "chineseeeg2_littleprince": PURPLE,
               "Pallier2025": BLUE, "ChineseEEG2": PURPLE}
NAMES = {"pallier2025": "Pallier2025", "chineseeeg2_littleprince": "ChineseEEG-2",
         "Pallier2025": "Pallier2025", "ChineseEEG2": "ChineseEEG-2"}
for font in ("arial.ttf", "arialbd.ttf", "ariali.ttf", "arialbi.ttf"):
    font_manager.fontManager.addfont("C:/Windows/Fonts/" + font)

STYLE = {
    "font.family": "sans-serif", "font.sans-serif": ["Arial"],
    "font.size": 6.5, "axes.labelsize": 6.5, "axes.titlesize": 7,
    "xtick.labelsize": 6, "ytick.labelsize": 6, "legend.fontsize": 6,
    "axes.linewidth": .55, "lines.linewidth": 1.1,
    "text.color": INK, "axes.labelcolor": INK,
    "xtick.color": INK, "ytick.color": INK, "axes.edgecolor": "#9BAAB4",
    "axes.grid": False, "pdf.fonttype": 42, "ps.fonttype": 42,
    "svg.fonttype": "none", "mathtext.fontset": "dejavusans",
}

@contextmanager
def style():
    with mpl.rc_context(STYLE):
        yield

def figure(height_mm=130, width_mm=183):
    return plt.figure(figsize=(width_mm / 25.4, height_mm / 25.4), facecolor="white")

def panel(fig, x, y, letter, title):
    fig.text(x, y, letter, fontsize=8, weight="bold", color=INK)
    fig.text(x + .026, y, title, fontsize=7, weight="bold", color=INK)

def clean(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(length=2.5, width=.55, color="#9BAAB4")

def cjk(bold=False):
    return FontProperties(fname="C:/Windows/Fonts/" + ("msyhbd.ttc" if bold else "msyh.ttc"))

def verify_manifest(filename, required=()):
    manifest = json.loads((EXPORT / filename).read_text(encoding="utf-8"))
    exports = {Path(e["path"]).name: e for e in manifest.get("exports", [])}
    if not set(required) <= set(exports):
        raise ValueError(f"Missing formal exports in {filename}")
    for entry in exports.values():
        if sha256((PROJECT_ROOT / entry["path"]).read_bytes()).hexdigest() != entry["sha256"]:
            raise ValueError(f"Changed formal export: {entry['path']}")
    return manifest

def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    paths = []
    # Keep the font contract active at render time, not only at artist creation.
    with style():
        for ext in ("pdf", "svg", "png"):
            path = OUT / f"{name}.{ext}"
            fig.savefig(path, dpi=360, facecolor="white", bbox_inches=None)
            paths.append(path)
    plt.close(fig)
    return paths
