"""Two summary figures for the class-frequency (series I) experiments, from the cached m0 ODE runs.

1. ``R1_class_loss_*``: per-class loss ell_c(t) vs t (log-log), Adam (one beta2) and SGD side by side,
   classes coloured by pi_c, with slope -1 and -1/2 guides.
2. ``R2_halftime_*``: half-loss time t_c (ell_c = ell_c(0)/2) vs pi_c, Adam (one beta2) and SGD,
   with slope -1 and -1/2 guides, one panel per C.

Usage (from the repository root):
    venv/bin/python experiments/imbalance/analysis/class_rates.py [--kappa 1] [--alpha 1] [--beta2 0.99]
"""
from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

import figures as F  # style, palette and save()
import lib
from lib import I


def guides(ax, x0, y0, x, labels=True):
    """Slope -1 (dashed) and -1/2 (dash-dot) lines through (x0, y0)."""
    ax.plot(x, y0 * (x / x0) ** -1.0, color=F.MUTED, ls="--", lw=1.1,
            label="slope −1" if labels else None, zorder=5)
    ax.plot(x, y0 * (x / x0) ** -0.5, color=F.MUTED, ls="-.", lw=1.1,
            label="slope −½" if labels else None, zorder=5)


def fig_class_loss(kappa, alpha, b2, C):
    runs = [("adam", b2), ("sgd", 0.0)]
    curs = [lib.curves("m0", o, kappa, C, alpha, b) for o, b in runs]
    if any(c is None for c in curs):
        print(f"skip R1 C={C}: missing run"); return
    pi = curs[0]["priors"]
    norm = LogNorm(pi.min(), pi.max())
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
    for ax, (o, b), cur in zip(axes, runs, curs):
        t, L = cur["time"], cur["loss"]
        k = t > 0
        for c in range(C)[::-1]:  # common classes drawn on top
            ax.plot(t[k], L[k, c], color=F.CLASS_CMAP(norm(pi[c])), lw=1.2)
        # guides over the last two decades, anchored just above the rarest class at T
        T = t[k][-1]
        x = np.geomspace(T / 100, T, 50)
        guides(ax, T, 2.5 * L[-1].max(), x, labels=ax is axes[0])
        ax.set(xscale="log", yscale="log", xlabel="t", title=F.name(o, b))
        F.polish_axis(ax)
    axes[0].set_ylabel("per-class loss ℓ_c(t)")
    axes[0].legend(loc="lower left")
    sm = plt.cm.ScalarMappable(norm=norm, cmap=F.CLASS_CMAP)
    fig.colorbar(sm, ax=axes.tolist(), label="π_c", pad=0.01)
    fig.suptitle(f"C = {C}, K = {kappa:g}·I, α = {alpha:g}", y=1.02, fontsize=13)
    F.save(fig, f"R1_class_loss_k{kappa:g}_a{alpha:g}_b2{b2:g}_C{C}")


def fig_halftime(kappa, alpha, b2):
    fig, axes = plt.subplots(1, len(I.CLASSES), figsize=(5 * len(I.CLASSES), 4.4))
    for ax, C in zip(np.atleast_1d(axes), I.CLASSES):
        anchors = {}
        for o, b in [("adam", b2), ("sgd", 0.0)]:
            cur = lib.curves("m0", o, kappa, C, alpha, b)
            if cur is None:
                continue
            t, L, pi = cur["time"], cur["loss"], cur["priors"]
            tc = lib.level_times(t, L, 0.5)
            ok = np.isfinite(tc)
            col = F.color(o, b)
            ax.plot(pi[ok], tc[ok], "o-", color=col, ms=4, lw=1, label=F.name(o, b))
            if (~ok).any():  # not halved by the horizon
                ax.plot(pi[~ok], np.full((~ok).sum(), t[-1]), "^", mfc="none", color=col, ms=6)
            anchors[o] = (pi, tc)
        if anchors:
            pi = next(iter(anchors.values()))[0]
            x = np.geomspace(pi.min(), pi.max(), 50)
            # each guide anchored 30% below its optimizer's most common class
            if "sgd" in anchors:
                ax.plot(x, 0.7 * anchors["sgd"][1][0] * (x / pi[0]) ** -1.0, color=F.MUTED,
                        ls="--", lw=1.1, label="slope −1")
            if "adam" in anchors:
                ax.plot(x, 0.7 * anchors["adam"][1][0] * (x / pi[0]) ** -0.5, color=F.MUTED,
                        ls="-.", lw=1.1, label="slope −½")
        ax.set(xscale="log", yscale="log", xlabel="π_c", title=f"C = {C}")
        ax.xaxis.set_minor_formatter(plt.NullFormatter())
        F.polish_axis(ax)
    np.atleast_1d(axes)[0].set_ylabel("t_c  (ℓ_c = ℓ_c(0)/2)")
    np.atleast_1d(axes)[0].legend(loc="lower left")
    fig.suptitle(f"K = {kappa:g}·I, α = {alpha:g}; triangles: not halved by T", y=1.02, fontsize=13)
    F.save(fig, f"R2_halftime_k{kappa:g}_a{alpha:g}_b2{b2:g}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--kappa", type=float, nargs="+", default=[1.0])
    p.add_argument("--alpha", type=float, nargs="+", default=[1.0])
    p.add_argument("--beta2", type=float, default=0.99)
    a = p.parse_args()
    for kappa in a.kappa:
        for alpha in a.alpha:
            for C in I.CLASSES:
                fig_class_loss(kappa, alpha, a.beta2, C)
            fig_halftime(kappa, alpha, a.beta2)
