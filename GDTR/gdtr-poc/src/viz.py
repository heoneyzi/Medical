"""Publication-style visualization helpers (Phase 0).

- setup_publication_style: rcParams tuned for 8 pt body / 10 pt panel labels,
  Wong 2011 colorblind-safe palette.
- save_figure: writes both .pdf (vector) and .png (300 dpi) at a given base path.
- add_significance_annotation: simple asterisk bracket between two x positions.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict

import matplotlib as mpl
import matplotlib.pyplot as plt

log = logging.getLogger(__name__)


# Wong 2011 colorblind-safe palette (8 colors, RGB hex)
WONG_PALETTE: Dict[str, str] = {
    "black":         "#000000",
    "orange":        "#E69F00",
    "sky_blue":      "#56B4E9",
    "bluish_green":  "#009E73",
    "yellow":        "#F0E442",
    "blue":          "#0072B2",
    "vermillion":    "#D55E00",
    "reddish_purple":"#CC79A7",
}


def setup_publication_style() -> Dict[str, str]:
    """Set matplotlib rcParams for publication-style figures.

    Returns:
        Wong 2011 palette dict (color name -> hex).
    """
    mpl.rcParams.update({
        "font.family":      ["DejaVu Sans"],   # Helvetica/Arial fallback
        "font.size":        8.0,
        "axes.titlesize":   9.0,
        "axes.labelsize":   8.0,
        "xtick.labelsize":  7.5,
        "ytick.labelsize":  7.5,
        "legend.fontsize":  7.0,
        "figure.titlesize": 10.0,
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "axes.linewidth":   0.8,
        "xtick.major.width":0.8,
        "ytick.major.width":0.8,
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,
        "lines.linewidth":  1.2,
        "savefig.dpi":      300,
        "figure.dpi":       110,
        "pdf.fonttype":     42,   # TrueType (editable in Illustrator)
        "ps.fonttype":      42,
    })
    return dict(WONG_PALETTE)


def save_figure(fig, path: str | Path) -> None:
    """Save fig to {path}.pdf and {path}.png (300 dpi).

    Args:
        fig: matplotlib Figure.
        path: base path WITHOUT extension.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(path) + ".pdf", bbox_inches="tight")
    fig.savefig(str(path) + ".png", bbox_inches="tight", dpi=300)
    log.info("saved %s.{pdf,png}", path)


def add_significance_annotation(
    ax: plt.Axes,
    x1: float,
    x2: float,
    y: float,
    p: float,
    height: float = 0.02,
    color: str = "black",
) -> None:
    """Draw a horizontal bracket between (x1, y) and (x2, y) with asterisks.

    Args:
        ax: matplotlib Axes.
        x1, x2: x positions.
        y: bracket y position (data coords).
        p: p-value (drives asterisk count).
        height: tick height.
        color: line/text color.
    """
    if p < 1e-3:
        stars = "***"
    elif p < 1e-2:
        stars = "**"
    elif p < 5e-2:
        stars = "*"
    else:
        stars = "ns"
    ax.plot([x1, x1, x2, x2], [y, y + height, y + height, y],
            color=color, lw=0.8)
    ax.text((x1 + x2) / 2, y + height, stars,
            ha="center", va="bottom", color=color, fontsize=7.5)
