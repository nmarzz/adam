"""Figure of the gptkink learning-rate tuning (results/gptkink/tune/*.json).

(a) held-out loss after 1e8 tokens against peak lr, one line per batch size b;
(b) the same against lr / sqrt(b / 16384), i.e. in units of the square-root rule;
(c) best lr against b with a fitted power law and the sqrt rule anchored at b = 16384.

Usage (from the repository root; numpy + matplotlib):
    python3 experiments/gptkink_tune_plot.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from plot_style import apply_paper_style, polish_axis

ROOT = Path(__file__).resolve().parents[1]
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]  # fixed categorical order (b ascending)
B0 = 16384


def load():
    rows = {}
    for f in sorted((ROOT / "results" / "gptkink" / "tune").glob("*.json")):
        d = json.loads(f.read_text())
        rows.setdefault(d["b"], []).append((d["lr"], d["final_loss"]))
    return {b: sorted(r) for b, r in sorted(rows.items())}


def main():
    apply_paper_style()
    plt.rcParams.update({"lines.linewidth": 2.0})
    rows = load()
    bs = list(rows)
    col = dict(zip(bs, COLORS))
    best = {b: min((x for x in r if x[1] is not None), key=lambda x: x[1]) for b, r in rows.items()}
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.6))

    for ax, scaled in ((axes[0], False), (axes[1], True)):
        for b, r in rows.items():
            lr = np.array([x[0] for x in r]); loss = np.array([x[1] for x in r], float)
            xs = lr / np.sqrt(b / B0) if scaled else lr
            ax.semilogx(xs, loss, "-o", ms=8, color=col[b], label=f"b = {b // 1024}k")
            bx = best[b][0] / np.sqrt(b / B0) if scaled else best[b][0]
            ax.plot(bx, best[b][1], "o", ms=15, mfc="none", mec=col[b], mew=2)
            ax.annotate(f"{b // 1024}k", (xs[-1], loss[-1]), xytext=(12, 0), textcoords="offset points",
                        va="center", fontsize=13, color="0.25")
        ax.set_xlim(right=ax.get_xlim()[1] * 2.2)
        ax.set_ylabel("held-out loss after $10^8$ tokens")
        polish_axis(ax)
    axes[0].set_xlabel(r"peak learning rate $\eta$")
    axes[0].set_title("(a) loss vs learning rate", fontsize=16)
    axes[0].legend(loc="upper right", fontsize=12, title="ring = best", title_fontsize=11)
    axes[1].set_xlabel(r"$\eta\,/\sqrt{b/16\mathrm{k}}$  (square-root rule units)")
    axes[1].set_title(r"(b) rescaled by $\sqrt{b}$", fontsize=16)

    ax = axes[2]
    bb = np.array(bs, float); lb = np.array([best[b][0] for b in bs])
    p = np.polyfit(np.log(bb), np.log(lb), 1)
    grid = np.geomspace(bb.min() / 1.5, bb.max() * 1.5, 50)
    ax.loglog(grid, np.exp(np.polyval(p, np.log(grid))), "-", color="0.35", lw=1.6,
              label=rf"fit $\eta^\star \propto b^{{{p[0]:.2f}}}$")
    ax.loglog(grid, 1e-3 * np.sqrt(grid / B0), "--", color="0.6", lw=1.6, label=r"$\sqrt{b}$ rule (grid centre)")
    for b in bs:
        ax.loglog(b, best[b][0], "o", ms=11, color=col[b])
        ax.annotate(f"{best[b][0]:.2g}", (b, best[b][0]), xytext=(0, -22), textcoords="offset points",
                    ha="center", fontsize=13, color="0.25")
    ax.set_xlabel("batch size b (tokens per step)")
    ax.set_ylabel(r"best peak learning rate $\eta^\star$")
    ax.set_title("(c) best learning rate vs b", fontsize=16)
    ax.legend(loc="upper left", fontsize=12)
    polish_axis(ax)

    fig.tight_layout()
    out = ROOT / "figs" / "gptkink"
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "tuning.pdf", metadata={"CreationDate": None})
    fig.savefig(out / "tuning.png", dpi=150)
    print(f"slope {p[0]:.3f}; saved {out / 'tuning.png'}")
    for b in bs:
        print(f"b={b:>6}: " + "  ".join(f"{lr:.3g}:{l:.3f}" for lr, l in rows[b]) + f"   best {best[b][0]:.3g}")


if __name__ == "__main__":
    main()
