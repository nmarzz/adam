"""E2 (exponents) analysis: local exponents of the risk in the window where the frontier c*(t)
is between 2 and C/4, against 2(alpha-1)/alpha (Adam/PSGD window) and (alpha-1)/alpha (SGD,
rare-event regime). numpy only; run from this directory."""
import os

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
OUT = REPO + "/figs/experiments/zipf/ode"
C, T, SAFETY = int(os.environ.get("C", "64")), 1000.0, float(os.environ.get("SAFETY", "1"))
HI = C // 4                                                       # fit window c* in [2, C/4]
ETA = {"adam": 0.3, "psgd": 0.3, "sgd": 3.0}
KAPPA = (np.pi / 2) * ((1 - 0.9 / np.sqrt(0.99)) / 0.1) ** 2      # 1.43 for beta2 = 0.99


def priors(alpha):
    p = np.arange(1, C + 1, dtype=float) ** -alpha
    return p / p.sum()


def load(mode, alpha, b):
    opt = {"adam": f"b20.99_b{b}_e1e-08", "psgd": f"b{b}_e1e-08", "sgd": f"b{b}"}[mode]
    p = f"{OUT}/{mode}_C{C}_a{alpha:g}_{opt}_eta{ETA[mode]:g}_T{T:g}_s{SAFETY:g}.npz"
    return dict(np.load(p)) if os.path.exists(p) else None


def t_learn(t, L):
    out = []
    for c in range(L.shape[1]):
        k = np.flatnonzero(L[:, c] <= L[0, c] / 2)
        out.append(t[k[0]] if len(k) else np.inf)
    return np.array(out)


def fit_exponent(t, R, mask):
    m = mask & (t > 0)
    if m.sum() < 4:
        return np.nan, 0
    return -np.polyfit(np.log(t[m]), np.log(R[m]), 1)[0], m.sum()


print(f"{'run':22s} {'R(T)':>7s} {'learned':>8s} | {'fit window c* in [2,C/4]':>24s} {'exp':>6s} | "
      f"pred Adam-window 2(a-1)/a, SGD (a-1)/a | {'c* in [2,6]':>11s} {'c* in [6,16]':>12s}")
for alpha in (1.25, 1.5, 2.0, 3.0):
    pi = priors(alpha)
    for b in (16, 1):
        for mode in ("adam", "psgd", "sgd"):
            r = load(mode, alpha, b)
            if r is None:
                continue
            t, L = r["time"], r["loss"]
            R = L @ pi
            tc = t_learn(t, L)
            cstar = np.array([(tc <= x).sum() for x in t])
            win = (cstar >= 2) & (cstar <= HI)
            e, n = fit_exponent(t, R, win)
            e1, _ = fit_exponent(t, R, (cstar >= 2) & (cstar <= 6))
            e2, _ = fit_exponent(t, R, (cstar >= 6) & (cstar <= HI))
            tw = t[win]
            span = f"t {tw.min():.2g}-{tw.max():.2g}" if len(tw) else "-"
            print(f"{mode:5s} a={alpha:<5g} b={b:<3d} {R[-1]:7.4f} {np.isfinite(tc).sum():5d}/64 | "
                  f"{span:>24s} {e:6.2f} | {2 * (alpha - 1) / alpha:5.2f} {(alpha - 1) / alpha:5.2f}"
                  f"{'':>27s} | {e1:11.2f} {e2:12.2f}")
    kink = KAPPA * 0.01 / np.array([1, 16])
    cls = [(pi > k).sum() for k in kink]
    print(f"   alpha={alpha:g}: classes inside the beta2 window (pi > kappa(1-beta2)/b): "
          f"b=1: {cls[0]}, b=16: {cls[1]}")


# ----------------------------------------------------------------------------- finite-C predictions
# If t_c ~ pi_c^-s and R(t) ~ sum_{c > c*(t)} pi_c ell_c(0), then c* ~ t^(1/(alpha s)) and the local
# exponent is (1/(alpha s)) * (-d log tail / d log c*), with the tail truncated at C. Averaged over
# c* in [2, 16] (log-uniform). The measured frontier exponent gamma = d log c* / d log t is fitted on
# the same window.
def tail_slope(alpha, cstar):
    """-d log tail / d log c* for the continuum tail int_{c*+1/2}^{C+1/2} k^-alpha dk."""
    x, top = cstar + 0.5, C + 0.5
    return cstar * x ** -alpha * (alpha - 1) / (x ** (1 - alpha) - top ** (1 - alpha))


print(f"\nfinite-C predictions (mean over c* in [2,{HI}]) and measured frontier exponent gamma = dlog c*/dlog t:")
grid = np.geomspace(2.0, float(HI), 25)
for alpha in (1.25, 1.5, 2.0, 3.0):
    ts = np.mean([tail_slope(alpha, x) for x in grid])
    line = [f"alpha={alpha:g}: -dlog tail/dlog c* = {ts:.2f}; predicted exponent window (s=1/2) {2 * ts / alpha:.2f}, SGD (s=1) {ts / alpha:.2f}"]
    for b in (16, 1):
        for mode in ("adam", "psgd", "sgd"):
            r = load(mode, alpha, b)
            if r is None:
                continue
            t, L = r["time"], r["loss"]
            tc = t_learn(t, L)
            cstar = np.array([(tc <= x).sum() for x in t])
            win = (cstar >= 2) & (cstar <= HI)
            if win.sum() >= 4:
                gamma = np.polyfit(np.log(t[win]), np.log(cstar[win]), 1)[0]
                line.append(f"{mode} b={b}: gamma {gamma:.2f} (-> s = {1 / (alpha * gamma):.2f})")
    print("  " + "; ".join(line))
