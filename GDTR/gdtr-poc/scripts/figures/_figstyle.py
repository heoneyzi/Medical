"""Shared publication-style helpers for figures_v2 (ICML manuscript)."""
from __future__ import annotations

from pathlib import Path
from typing import Dict

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt


# Colorblind-safe palette (Wong 2011), aligned with existing Phase 0-5 figures
WONG_PALETTE: Dict[str, str] = {
    "black":          "#000000",
    "orange":         "#E69F00",
    "sky_blue":       "#56B4E9",
    "bluish_green":   "#009E73",
    "yellow":         "#F0E442",
    "blue":           "#0072B2",
    "vermillion":     "#D55E00",
    "reddish_purple": "#CC79A7",
}

# Named ordered colors for consistent assignment across figures
COLORS = {
    "dD_cos":     WONG_PALETTE["blue"],
    "dD_jsd":     WONG_PALETTE["sky_blue"],
    "evo2":       WONG_PALETTE["orange"],
    "alphamis":   WONG_PALETTE["reddish_purple"],
    "ensemble":   WONG_PALETTE["bluish_green"],
    "cadd":       WONG_PALETTE["vermillion"],
    "control":    "#888888",
    "highlight":  WONG_PALETTE["vermillion"],
}


def setup() -> Dict[str, str]:
    """Apply rcParams for ICML manuscript figures and return palette."""
    mpl.rcParams.update({
        "font.family":       ["DejaVu Sans"],
        "font.size":         9.0,
        "axes.titlesize":    11.0,
        "axes.labelsize":    9.0,
        "xtick.labelsize":   8.0,
        "ytick.labelsize":   8.0,
        "legend.fontsize":   7.5,
        "figure.titlesize":  11.0,
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "axes.linewidth":    0.8,
        "xtick.direction":   "out",
        "ytick.direction":   "out",
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.major.size":  3.0,
        "ytick.major.size":  3.0,
        "lines.linewidth":   1.2,
        "savefig.dpi":       300,
        "figure.dpi":        110,
        "pdf.fonttype":      42,
        "ps.fonttype":       42,
        "axes.grid":         False,
    })
    return dict(WONG_PALETTE)


def grid_y(ax) -> None:
    """Apply light gray major-y gridlines."""
    ax.yaxis.grid(True, color="#dddddd", linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)


def label_panel(ax, label: str, x: float = -0.13, y: float = 1.05) -> None:
    """Add bold panel letter (a)/(b)/... at upper-left of axes."""
    ax.text(x, y, label, transform=ax.transAxes, fontsize=12,
            fontweight="bold", ha="left", va="bottom")


def save(fig, base_path: str | Path) -> None:
    """Save figure as both .pdf (vector) and .png (300 dpi)."""
    base = Path(base_path)
    base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(base) + ".pdf", bbox_inches="tight")
    fig.savefig(str(base) + ".png", bbox_inches="tight", dpi=300)
    plt.close(fig)
