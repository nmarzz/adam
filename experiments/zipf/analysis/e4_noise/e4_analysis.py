"""E4 (noise / eta sweep) analysis: numbers for the log and the mechanism figure.

Reads the cached ODE runs ``figs/experiments/zipf/ode/*.npz`` written by
``zipf_scaling.py noise`` (C = 64, alpha = 2, b = 1, eta T fixed), and, if present,
``simcheck_*.npz`` (direct Adam runs). numpy only. Run from this directory:
``../../../../real/bin/python e4_analysis.py``.

Quantities (K = Id, means r q_c orthonormal):
* margin of class c: gap_c = O_cc - max_{k != c} O_kc (logit gap on the class mean);
  its z-score uses the variance of the logit difference, B_cc + B_kk - 2 B_ck;
* noise energy N = B - O O^T / r^2 (part of theta outside the span of the means);
  in the isotropic ODE it obeys dN/dt = -eta Sym(N phi) + eta^2 sigma exactly;
* shift-invariant stationarity: largest change of all logit gaps O_cc - O_kc over the
  last half-decade of the run, relative to their size.
"""
import glob
import json
import os

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/tmp/adam-matplotlib")
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
ODE = REPO + "/figs/experiments/zipf/ode"
OUT = REPO + "/figs/experiments/zipf/analysis/e4_noise"
os.makedirs(OUT, exist_ok=True)
R2, C, ALPHA, SAFETY = 25.0, 64, 2.0, 1.0
PROG, PROG_SGD = 900.0, 9000.0
ETAS, SGD_ETAS, EPSS = (0.03, 0.1, 0.3, 1.0), (0.3, 1.0, 3.0, 10.0), (1e-8, 1e-4)
PI = np.arange(1, C + 1, dtype=float) ** -ALPHA
PI /= PI.sum()
SHOW = (7, 15, 31, 63)


def path(mode, eta, eps):
    T = (PROG_SGD if mode == "sgd" else PROG) / eta
    opt = {"adam": f"b20.99_b1_e{eps:g}", "psgd": f"b1_e{eps:g}", "sgd": "b1"}[mode]
    return f"{ODE}/{mode}_C{C}_a{ALPHA:g}_{opt}_eta{eta:g}_T{float(np.round(T, 6)):g}_s{SAFETY:g}.npz"


def load(mode, eta, eps=1e-8):
    p = path(mode, eta, eps)
    return dict(np.load(p)) if os.path.exists(p) else None


def margins(B, O):
    """gap_c, z_c for one state (B, O)."""
    gap, z = np.empty(C), np.empty(C)
    for c in range(C):
        others = np.delete(np.arange(C), c)
        g = O[c, c] - O[others, c]
        var = B[c, c] + B[others, others] - 2 * B[c, others]
        zz = g / np.sqrt(np.maximum(var, 1e-12))
        j = np.argmin(zz)
        gap[c], z[c] = g.min(), zz[j]
    return gap, z


def learned(t, L):
    return np.array([np.any(L[:, c] <= L[0, c] / 2) for c in range(C)])


def stationarity(t, O):
    """max |change| of the logit gaps O_cc - O_kc over t in [T/sqrt(10), T], relative."""
    k0 = np.argmin(abs(t - t[-1] / np.sqrt(10)))
    G = lambda o: np.diagonal(o)[None, :] - o
    a, b = G(O[k0]), G(O[-1])
    return np.abs(b - a).max() / np.abs(b).max()


runs = {}
for eps in EPSS:
    for eta in ETAS:
        for mode in ("adam", "psgd"):
            r = load(mode, eta, eps)
            if r is not None:
                runs[(mode, eta, eps)] = r
for eta in SGD_ETAS:
    r = load("sgd", eta)
    if r is not None:
        runs[("sgd", eta, 0.0)] = r

summary = {}
print(f"{'run':26s} {'T':>6s} {'R(T)':>7s} {'learned':>7s} {'stat':>7s} | {'B_cc(T)':>7s} {'N/B':>5s} | "
      f"{'gap_c(T) c=8,16,32,64':>26s} | {'z_c(T)':>22s} | {'phi_cc(T) c=8,64':>16s} | gap~pi^s")
for (mode, eta, eps), r in sorted(runs.items(), key=lambda kv: (kv[0][0], kv[0][2], kv[0][1])):
    t, B, O, L, co = r["time"], r["B"], r["O"], r["loss"], r["coef"]
    gap, z = margins(B[-1], O[-1])
    Nd = np.diagonal(B[-1]) - (O[-1] ** 2).sum(1) / R2
    tail = np.arange(C) >= 15
    slope = np.polyfit(np.log(PI[tail]), np.log(np.maximum(gap[tail], 1e-6)), 1)[0]
    st = stationarity(t, O)
    lab = f"{mode} eta={eta:g} eps={eps:g}"
    summary[lab] = dict(T=float(t[-1]), R=float(L[-1] @ PI), learned=int(learned(t, L).sum()), stationarity=float(st),
                        B_max=float(np.diagonal(B[-1]).max()), noise_frac=float((Nd / np.diagonal(B[-1])).mean()),
                        gap=gap[list(SHOW)].tolist(), z=z[list(SHOW)].tolist(), gap_slope=float(slope),
                        phi=[float(co[-1, 0, 7]), float(co[-1, 0, 63])])
    s = summary[lab]
    print(f"{lab:26s} {t[-1]:6.0f} {s['R']:7.4f} {s['learned']:5d}/64 {st:7.1e} | {s['B_max']:7.1f} {s['noise_frac']:5.2f} | "
          + " ".join(f"{g:6.2f}" for g in s["gap"]) + " | " + " ".join(f"{v:5.2f}" for v in s["z"]) + " | "
          + " ".join(f"{v:7.4f}" for v in s["phi"]) + f" | {slope:+.2f}")

print("\nEqual-t comparison (Adam and PSGD; tail = classes 17-64, head = 1-16; loss normalized by prior mass):")
tail = np.arange(C) >= 16
eq = {}
for t0 in (30.0, 100.0, 300.0, 900.0):
    rows = []
    for (mode, eta, eps), r in runs.items():
        if mode == "sgd" or r["time"][-1] < t0 * 0.999:
            continue
        L = np.array([np.interp(np.log(t0), np.log(r["time"][1:]), r["loss"][1:, c]) for c in range(C)])
        rows.append((f"{mode} eta={eta:g} eps={eps:g}", L @ PI, L[tail] @ PI[tail] / PI[tail].sum(),
                     L[~tail] @ PI[~tail] / PI[~tail].sum()))
    if rows:
        best = {k: min(rows, key=lambda x: x[i])[0] for k, i in (("risk", 1), ("tail", 2), ("head", 3))}
        eq[t0] = dict(best=best, rows=rows)
        print(f"  t={t0:5g}: best risk {best['risk']}, best tail {best['tail']}, best head {best['head']}")

sims = sorted(glob.glob(REPO + "/figs/experiments/zipf/simcheck_*.npz"))
for f in sims:
    d = np.load(f)
    t, B, O, L = d["time"], d["B"], d["O"], d["loss"]
    print(f"\n{os.path.basename(f)}: {B.shape[0]} runs, T = {t[-1]:g}")
    for tt in (10, 100, 300, 1000, 3000):
        k = np.argmin(abs(t - tt))
        if abs(t[k] - tt) > 0.3 * tt:
            continue
        gz = np.array([margins(B[i, k], O[i, k]) for i in range(B.shape[0])])   # (runs, 2, C)
        print(f"  t={t[k]:7.1f}: gap_c c=8,16,32,64 mean over runs " + " ".join(f"{gz[:, 0, c].mean():6.2f}" for c in SHOW)
              + "   z_c " + " ".join(f"{gz[:, 1, c].mean():5.2f}" for c in SHOW)
              + f"   mean B_cc {np.diagonal(B[:, k], axis1=-2, axis2=-1).mean():7.1f}"
              + f"   learned {np.mean([learned(t[:k + 1], L[i, :k + 1]).sum() for i in range(B.shape[0])]):.1f}")

json.dump(dict(runs=summary, equal_t={str(k): v["best"] for k, v in eq.items()}),
          open(OUT + "/e4_numbers.json", "w"), indent=1)

# ----------------------------------------------------------------------------- figure
fig, axes = plt.subplots(1, 4, figsize=(26, 6), constrained_layout=True)
styles = {("adam", 1e-8): "-", ("psgd", 1e-8): "--", ("adam", 1e-4): ":"}
cols = dict(zip(SHOW, plt.cm.plasma(np.linspace(0.05, 0.8, len(SHOW)))))
for (mode, eps), ls in styles.items():
    r = runs.get((mode, 0.3, eps))
    if r is None:
        continue
    t = r["time"]
    gz = np.array([margins(r["B"][k], r["O"][k]) for k in range(len(t))])
    for c in SHOW:
        axes[0].plot(t[1:], np.maximum(gz[1:, 0, c], 1e-3), ls, color=cols[c], lw=1.8)
        axes[1].plot(t[1:], r["coef"][1:, 0, c], ls, color=cols[c], lw=1.8)
    axes[0].plot([], [], ls, color="0.4", label=f"{'Adam' if mode == 'adam' else 'PSGD'} $\\epsilon={eps:g}$")
for c in SHOW:
    axes[0].plot([], [], "-", color=cols[c], label=f"$c={c + 1}$")
for ax, lab, title in ((axes[0], r"$O_{cc}-\max_{k\neq c}O_{kc}$", r"Margin on the class mean ($\eta=0.3$)"),
                       (axes[1], r"$\phi_{cc}$", r"Normalized curvature $\phi_{cc}$")):
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("$t$")
    ax.set_ylabel(lab)
    ax.set_title(title)
axes[0].legend(frameon=False, fontsize=11)
ax = axes[2]
for (mode, eta, eps), r in runs.items():
    if mode == "sgd" or eps != 1e-8:
        continue
    gap, _ = margins(r["B"][-1], r["O"][-1])
    col = plt.cm.viridis(np.searchsorted(ETAS, eta) / len(ETAS))
    ax.plot(PI, np.maximum(gap, 1e-3), "o" if mode == "adam" else "^", color=col, ms=4,
            label=f"{'Adam' if mode == 'adam' else 'PSGD'} $\\eta={eta:g}$")
x = np.array([PI[15], PI[-1]])
ax.plot(x, 3 * x / x[0], "k:", lw=1, label="slope 1")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel(r"$\pi_c$")
ax.set_ylabel("margin at the end of the run")
ax.set_title(r"End-of-run margins ($\eta T=900$, $\epsilon=10^{-8}$)")
ax.legend(frameon=False, fontsize=9, ncol=2)
ax = axes[3]
for (mode, eta, eps), r in runs.items():
    if mode != "adam":
        continue
    t = r["time"][1:]
    G = np.array([np.diagonal(o)[None, :] - o for o in r["O"][1:]])
    change = np.abs(np.diff(G, axis=0)).max((1, 2)) / np.abs(G[1:]).max((1, 2)) / np.diff(np.log(t))
    ax.plot(t[1:], np.maximum(change, 1e-9), "-" if eps == 1e-8 else ":",
            color=plt.cm.viridis(np.searchsorted(ETAS, eta) / len(ETAS)), lw=1.6,
            label=f"$\\eta={eta:g}$, $\\epsilon={eps:g}$")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("$t$")
ax.set_ylabel(r"$\max|\Delta(O_{cc}-O_{kc})|\,/\,\max|O_{cc}-O_{kc}|$ per unit $\log t$")
ax.set_title("Shift-invariant state: rate of change (Adam)")
ax.legend(frameon=False, fontsize=9)
for ax in axes:
    ax.grid(color="0.85", lw=0.8)
fig.savefig(OUT + "/e4_mechanism.png", dpi=110)
print("\nsaved", OUT + "/e4_mechanism.png")
