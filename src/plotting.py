from __future__ import annotations

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

BG = "#0d1117"
PANEL = "#141b22"
GRID = "#263142"
TEXT = "#e6edf3"
MUTED = "#8b98a5"

GREEN_PRIMARY = "#00e676"
GREEN_DARK = "#1b5e20"
GREEN_MID = "#2e7d32"
TEAL = "#00bfa5"
AMBER = "#ffb300"
RED = "#ff5252"

PALETTE = [GREEN_PRIMARY, TEAL, "#64dd17", AMBER, "#40c4ff", RED, "#b388ff"]


def apply_theme():
    mpl.rcParams.update(
        {
            "figure.facecolor": BG,
            "axes.facecolor": PANEL,
            "savefig.facecolor": BG,
            "axes.edgecolor": GRID,
            "axes.labelcolor": TEXT,
            "axes.titlecolor": TEXT,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.alpha": 0.6,
            "grid.linewidth": 0.6,
            "text.color": TEXT,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "legend.facecolor": PANEL,
            "legend.edgecolor": GRID,
            "legend.labelcolor": TEXT,
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.titlesize": 14,
            "axes.titleweight": "bold",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 130,
            "savefig.dpi": 160,
            "savefig.bbox": "tight",
        }
    )


def style_axes(ax, title=None, xlabel=None, ylabel=None):
    if title:
        ax.set_title(title, pad=12)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    for spine in ax.spines.values():
        spine.set_color(GRID)
    return ax


def gradient_fill(ax, x, y, color=GREEN_PRIMARY, alpha_top=0.35):
    """Fill under a line with a soft vertical gradient for a modern look."""
    z = np.linspace(0, 1, 100).reshape(-1, 1)
    z = np.tile(z, (1, len(x)))
    xmin, xmax = np.min(x), np.max(x)
    im = ax.imshow(
        z,
        aspect="auto",
        extent=[xmin, xmax, 0, np.max(y) * 1.05 if np.max(y) > 0 else 1],
        origin="lower",
        cmap=_green_fade_cmap(color),
        alpha=alpha_top,
        zorder=1,
    )
    clip_path = ax.fill_between(x, y, 0, color="none")
    im.set_clip_path(clip_path.get_paths()[0], transform=ax.transData)
    return im


def _green_fade_cmap(color):
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list("green_fade", [BG, color])
