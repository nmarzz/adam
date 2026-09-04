"""Shared publication styling for experiment figures."""
from __future__ import annotations

import matplotlib.pyplot as plt


def apply_paper_style():
    """Set readable defaults for figures placed at paper-column width."""
    plt.rcParams.update({
        "font.size": 17,
        "axes.titlesize": 20,
        "axes.labelsize": 18,
        "axes.linewidth": 1.25,
        "axes.titlepad": 12,
        "axes.labelpad": 8,
        "xtick.labelsize": 15,
        "ytick.labelsize": 15,
        "xtick.major.size": 6,
        "ytick.major.size": 6,
        "xtick.major.width": 1.15,
        "ytick.major.width": 1.15,
        "legend.fontsize": 14,
        "legend.handlelength": 2.5,
        "legend.handletextpad": 0.7,
        "figure.titlesize": 22,
        "figure.dpi": 130,
        "savefig.dpi": 300,
        "lines.linewidth": 2.8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def polish_axis(axis, *, grid=True):
    """Apply consistent ticks, spines, and a restrained background grid."""
    axis.tick_params(direction="out", top=False, right=False)
    for spine in axis.spines.values():
        spine.set_linewidth(1.25)
    if grid:
        axis.grid(color="0.82", linewidth=0.9, alpha=0.55)
        axis.set_axisbelow(True)
