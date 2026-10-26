"""论文图形的公共视觉样式。

设计原则：克制、精确、留白充足、信息层级清晰；优先矢量输出和一致的
排版、间距与视觉语义，避免无必要边框、3D、装饰性渐变和彩虹配色。
在实际可行时，图形应当在灰度环境中仍然可读。
"""

from pathlib import Path
from typing import Iterable


WINDOWS_FONTS_DIR = Path("C:/Windows/Fonts")
PUBLICATION_PALETTE = {
    "ink": "#26343D",
    "muted": "#586873",
    "rule": "#D5DDE2",
    "blue": "#356F99",
    "teal": "#287F82",
    "orange": "#B86A50",
    "gray": "#97A4AE",
    "purple": "#86699B",
}
DATASET_DISPLAY_NAMES = {
    "pallier2025": "Pallier2025",
    "chineseeeg2_littleprince": "ChineseEEG-2",
    "Pallier2025": "Pallier2025",
    "ChineseEEG2": "ChineseEEG-2",
}
DATASET_COLORS = {
    "pallier2025": PUBLICATION_PALETTE["blue"],
    "chineseeeg2_littleprince": PUBLICATION_PALETTE["purple"],
    "Pallier2025": PUBLICATION_PALETTE["blue"],
    "ChineseEEG2": PUBLICATION_PALETTE["purple"],
}


def register_system_fonts() -> None:
    """Safely register Windows Arial and Microsoft YaHei fonts when present on disk."""
    from matplotlib import font_manager

    for font_name in ("arial.ttf", "arialbd.ttf", "ariali.ttf", "arialbi.ttf", "msyh.ttc", "msyhbd.ttc"):
        font_path = WINDOWS_FONTS_DIR / font_name
        if font_path.is_file():
            try:
                font_manager.fontManager.addfont(str(font_path))
            except Exception:
                pass


def cjk_font_properties(bold: bool = False):
    """Return FontProperties for CJK text with cross-platform fallback when Windows fonts are absent."""
    from matplotlib.font_manager import FontProperties

    font_path = WINDOWS_FONTS_DIR / ("msyhbd.ttc" if bold else "msyh.ttc")
    if font_path.is_file():
        return FontProperties(fname=str(font_path))
    return FontProperties(
        family=["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "PingFang SC", "DejaVu Sans"],
        weight="bold" if bold else "normal",
    )


MODEL_LABELS = {"word": "Word", "context": "Context"}
MODEL_STYLES = {
    "word": {"color": "#3B6EA8", "marker": "o", "linestyle": "-", "hatch": ""},
    "context": {
        "color": "#D97706",
        "marker": "s",
        "linestyle": "--",
        "hatch": "//",
    },
}
CHANCE_COLOR = "#7A7A7A"
NEUTRAL_COLOR = "#8B8B8B"

# Stable semantic styles for the reusable context/revision figures. Existing
# Word/Context figures retain their published colour assignments.
CURVE_STYLES = {
    "word": {key: value for key, value in MODEL_STYLES["word"].items() if key != "hatch"},
    "context": {key: value for key, value in MODEL_STYLES["context"].items() if key != "hatch"},
    "local": {"color": "#168C8C", "marker": "^", "linestyle": "-"},
    "rolling": {"color": "#3B6EA8", "marker": "o", "linestyle": "-"},
    "offline_local": {"color": "#168C8C", "marker": "^", "linestyle": "-."},
    "full": {"color": "#D97706", "marker": "s", "linestyle": "--"},
}


def configure_matplotlib(profile: str = "legacy") -> None:
    """Apply legacy styling or the serif, light-grid reference profile."""
    import matplotlib as mpl

    if profile not in {"legacy", "reference"}:
        raise ValueError("profile must be 'legacy' or 'reference'.")
    register_system_fonts()
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 9,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "legend.frameon": False,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.04,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "axes.grid": False,
            "mathtext.fontset": "dejavusans",
        }
    )
    if profile == "reference":
        mpl.rcParams.update(
            {
                "font.family": "serif",
                "font.serif": ["STIXGeneral", "DejaVu Serif"],
                "mathtext.fontset": "stix",
                "axes.spines.top": True,
                "axes.spines.right": True,
                "axes.linewidth": 0.55,
                "axes.edgecolor": "#777777",
                "axes.facecolor": "white",
                "figure.facecolor": "white",
                "axes.grid": True,
                "axes.axisbelow": True,
                "grid.color": "#E5E5E5",
                "grid.linewidth": 0.45,
                "grid.alpha": 1.0,
                "xtick.direction": "in",
                "ytick.direction": "in",
                "xtick.major.width": 0.55,
                "ytick.major.width": 0.55,
                "lines.linewidth": 1.35,
                "lines.markersize": 4.0,
                "lines.markeredgewidth": 0.85,
            }
        )


def figure_size(width: str = "single", aspect_ratio: float = 0.62) -> tuple[float, float]:
    """按单栏或双栏宽度返回 Figure 尺寸（英寸）。"""
    widths = {"single": 3.5, "double": 7.2}
    if width not in widths:
        raise ValueError("width 只能是 'single' 或 'double'。")
    if aspect_ratio <= 0:
        raise ValueError("aspect_ratio 必须大于 0。")
    figure_width = widths[width]
    return figure_width, figure_width * aspect_ratio


def save_figure(
    figure,
    output_stem: str | Path,
    formats: Iterable[str] = ("pdf", "svg"),
    *,
    dpi: int = 300,
) -> tuple[Path, ...]:
    """用一致参数保存论文图，默认生成 PDF 和 SVG。"""
    import matplotlib as mpl

    stem = Path(output_stem)
    stem.parent.mkdir(parents=True, exist_ok=True)

    saved_paths = []
    for file_format in formats:
        normalized_format = file_format.lower().lstrip(".")
        output_path = stem.with_suffix(f".{normalized_format}")
        # Export settings must also apply after a renderer's local style context
        # has exited (SVG keeps text editable; PDF embeds TrueType fonts).
        with mpl.rc_context({"pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none"}):
            figure.savefig(output_path, format=normalized_format, dpi=dpi,
                           bbox_inches="tight", pad_inches=0.04)
        saved_paths.append(output_path)
    return tuple(saved_paths)
