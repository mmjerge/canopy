"""Shared publication-quality matplotlib style for the example figures.

Import from a demo with a small path shim (the demos live one level below ``examples/``):

    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from _plotstyle import set_style, save_figure, ci_band, PALETTE
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

# Okabe-Ito colorblind-safe palette.
PALETTE = {
    "blue": "#0072B2",
    "orange": "#E69F00",
    "green": "#009E73",
    "red": "#D55E00",
    "purple": "#CC79A7",
    "sky": "#56B4E9",
    "gray": "#7F7F7F",
}

# Figures are written here: a tracked, visible location ready for the LaTeX project.
FIGURE_DIR = Path(__file__).resolve().parents[1] / "paper" / "figures"


def set_style() -> None:
    """Apply a clean, serif, colorblind-friendly style suited to the paper."""
    mpl.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "font.family": "serif",
            "font.serif": ["STIXGeneral", "Times New Roman", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 11,
            "axes.titlesize": 11,
            "axes.labelsize": 11,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "grid.linewidth": 0.6,
            "lines.linewidth": 2.0,
            "lines.markersize": 6,
            "legend.frameon": False,
        }
    )


def save_figure(fig, name: str) -> Path:
    """Save ``fig`` as both vector PDF (for LaTeX) and PNG (for quick viewing)."""
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(FIGURE_DIR / f"{name}.{ext}")
    plt.close(fig)
    return FIGURE_DIR / f"{name}.pdf"


def ci_band(ax, x, samples, color, label=None, marker="o"):
    """Plot the mean over seeds with a shaded 95% confidence band.

    ``samples`` has shape ``(len(x), n_seeds)``: one column of per-seed measurements per
    x value. Returns the mean line.
    """
    samples = np.asarray(samples, dtype=float)
    mean = samples.mean(axis=1)
    n = samples.shape[1]
    sem = samples.std(axis=1, ddof=1) / np.sqrt(max(1, n))
    half = 1.96 * sem
    (line,) = ax.plot(x, mean, marker=marker, color=color, label=label)
    ax.fill_between(x, mean - half, mean + half, color=color, alpha=0.18, linewidth=0)
    return line
