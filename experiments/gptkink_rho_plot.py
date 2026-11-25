"""Own-event forcing of the gptkink probes against token events per batch.

Two event counts per optimizer step for token c (B sequences of L tokens, b = B L):
* b * pi_c   - expected own occurrences (targets) in the batch;
* B * rho_c  - expected number of sequences in the batch that contain c, with rho_c the
               fraction of L-token windows whose targets contain c (data/gptkink/rho_L256.npy).
Without burstiness rho_c ~ L pi_c and the two agree; L pi_c / rho_c is the mean number of
occurrences of c in a window that contains it.

Figure ``figs/gptkink/forcing_vs_events.png``: rows = probe checkpoints (0, 0.3, 1),
columns = x axis (b pi_c, B rho_c); y = own-event forcing normalized by its value at
pi_c = 1e-2. Second figure ``burstiness.png``: L pi_c / rho_c against pi_c.

Usage (repository root; numpy + matplotlib):
    python3 experiments/gptkink_rho_plot.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("FREQKINK_GAIN", "gain_own_sum")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import freqkink_analysis as A  # noqa: E402
from plot_style import apply_paper_style, polish_axis  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
COLORS = {1024: "#2a78d6", 4096: "#eb6834", 16384: "#1baf7a", 65536: "#eda100"}
MARK = {0.9: "o", 0.99: "s"}
FRACS = ("0", "0.3", "1")
L = 256


def binned(runs, rho):
    pi, edges, k = A.bins_for(runs[0]["train_counts"])
    out = []
    for r in runs:
        bc = A.bin_curves(r, edges, k)
        y = A.rates(r, bc, "Gslope")
        tc = r["test_counts"]
        rb = []
        for j in range(1, len(edges)):
            m = (k == j) & (tc > 0)
            if tc[m].sum() >= 50:
                rb.append(np.sum(tc[m] * rho[m]) / tc[m].sum())
        rb = np.array(rb)
        ok = np.isfinite(y) & (y > 0)
        x = bc["pi"][ok]
        y = y[ok] / np.exp(np.interp(np.log(1e-2), np.log(x), np.log(y[ok])))
        out.append(dict(b=r["cfg"]["b"], B=r["cfg"]["B"], beta2=r["cfg"]["head_beta2"], pi=x, rho=rb[ok], y=y))
    return out


def half_point(x, y):
    i = np.nonzero(y >= 0.5)[0][0]
    return float(np.exp(np.interp(np.log(0.5), np.log(y[i - 1:i + 1]), np.log(x[i - 1:i + 1]))))


def main():
    apply_paper_style()
    rho = np.load(ROOT / "data" / "gptkink" / f"rho_L{L}.npy")
    fig, axes = plt.subplots(len(FRACS), 2, figsize=(13.5, 4.6 * len(FRACS)), sharey="row")
    print(f"{'frac':>5} {'beta2':>5} {'b':>6}  half-point:  b*pi   B*rho   (forcing at 50% of its pi=1e-2 value)")
    for row, fr in zip(axes, FRACS):
        A.OUT = ROOT / "results" / "gptkink" / "probes" / f"frac_{fr}"
        data = binned(A.load_runs(), rho)
        for d in sorted(data, key=lambda d: (d["beta2"], d["b"])):
            for ax, x in ((row[0], d["b"] * d["pi"]), (row[1], d["B"] * d["rho"])):
                ax.loglog(x, d["y"], MARK[d["beta2"]] + "-", ms=4.5, lw=1.2, color=COLORS[d["b"]], alpha=0.9,
                          mfc="none" if d["beta2"] == 0.9 else COLORS[d["b"]])
            print(f"{fr:>5} {d['beta2']:>5} {d['b']:>6}  {half_point(d['b'] * d['pi'], d['y']):10.2f} "
                  f"{half_point(d['B'] * d['rho'], d['y']):7.2f}")
        for ax in row:
            ax.axvline(1.0, color="0.3", ls="--", lw=1.2)
            polish_axis(ax)
        row[0].set_ylabel(f"forcing / forcing($\\pi$=1e-2)\ncheckpoint {float(fr):.0%}")
    axes[-1, 0].set_xlabel(r"own occurrences per batch  $b\,\pi_c$")
    axes[-1, 1].set_xlabel(r"sequences per batch containing $c$  $B\,\rho_c$")
    axes[0, 0].set_title(r"x = $b\pi_c$ (tokens)", fontsize=16)
    axes[0, 1].set_title(r"x = $B\rho_c$ (independent sequences)", fontsize=16)
    handles = [plt.Line2D([], [], color=c, marker="o", ls="-", label=f"b = {b // 1024}k") for b, c in COLORS.items()]
    handles += [plt.Line2D([], [], color="0.3", marker=m, ls="", mfc="none" if v == 0.9 else "0.3",
                           label=rf"head $\beta_2$ = {v}") for v, m in MARK.items()]
    handles += [plt.Line2D([], [], color="0.3", ls="--", label="one event per batch")]
    axes[0, 0].legend(handles=handles, fontsize=11, loc="lower right", ncol=2)
    fig.tight_layout()
    out = ROOT / "figs" / "gptkink"
    fig.savefig(out / "forcing_vs_events.png", dpi=150)
    fig.savefig(out / "forcing_vs_events.pdf", metadata={"CreationDate": None})
    plt.close(fig)

    # burstiness: mean occurrences per containing window, by frequency bin
    counts = np.load(ROOT / "data" / "gptkink" / "train_counts.npy")
    pi = counts / counts.sum()
    ok = (counts > 0) & (rho > 0)
    u = np.log10(pi[ok]); burst = L * pi[ok] / rho[ok]
    edges = np.arange(np.floor(u.min()), np.ceil(u.max()) + 0.25, 0.25)
    k = np.digitize(u, edges)
    cen = [10 ** u[k == j].mean() for j in range(1, len(edges)) if (k == j).sum() >= 5]
    med = [np.median(burst[k == j]) for j in range(1, len(edges)) if (k == j).sum() >= 5]
    q = [np.quantile(burst[k == j], [0.25, 0.75]) for j in range(1, len(edges)) if (k == j).sum() >= 5]
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    ax.fill_between(cen, [a for a, _ in q], [b_ for _, b_ in q], color="#2a78d6", alpha=0.18, lw=0, label="interquartile")
    ax.semilogx(cen, med, "-o", ms=5, color="#2a78d6", label="median over tokens")
    ax.axhline(1, color="0.4", ls="--", lw=1.2, label="no burstiness")
    ax.set_xlabel(r"token frequency $\pi_c$")
    ax.set_ylabel(r"$L\pi_c/\rho_c$")
    ax.set_title("occurrences per window containing $c$", fontsize=14)
    ax.legend(fontsize=11)
    polish_axis(ax)
    fig.tight_layout()
    fig.savefig(out / "burstiness.png", dpi=150)
    fig.savefig(out / "burstiness.pdf", metadata={"CreationDate": None})
    print("burstiness (median L pi/rho) by decade:", {f"1e{int(np.floor(np.log10(c)))}": round(m, 2)
                                                      for c, m in zip(cen[::4], med[::4])})


if __name__ == "__main__":
    main()
