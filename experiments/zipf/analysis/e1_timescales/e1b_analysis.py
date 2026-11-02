"""E1b: per-class learning times t_c vs pi_c from figs/experiments/zipf/timescales.npz.

Run (repo root):  JAX_PLATFORMS=cpu real/bin/python <this file>
Reads timescales.npz (and rates.npz for the frozen-rate reference); writes only to the
scratch dir: e1b_numbers.json. Prints markdown tables. numpy only, < 10 s.
"""
import json
from math import comb, erf, sqrt

import numpy as np

from e1b_lib import (ADAM, BETA1, ETA_ADAM, REPO, SCR, SGD, SHORT, bpl_sharp, bpl_smooth,
                     first_crossing, interp_log, kappa_beta, kink_beta2, load, ols)

r = load()
pi, t = r["priors"], r["time"]
lpi = np.log(pi)
C = len(pi)
rates = dict(np.load(REPO + "/figs/experiments/zipf/rates.npz"))
NUM = {}


def tc(k):
    return r[k + "_tc"]


def local_slope(x_pi, y_t, half=0.2):
    """OLS slope of log t on log pi within +-half decades (>= 3 finite points); else
    the secant between the finite neighbours c-1 and c+1."""
    lp = np.log10(x_pi)
    out = np.full(len(x_pi), np.nan)
    fin = np.isfinite(y_t)
    for c in range(len(x_pi)):
        if not fin[c]:
            continue
        m = (np.abs(lp - lp[c]) <= half) & fin
        if m.sum() >= 3:
            out[c] = ols(np.log(x_pi[m]), np.log(y_t[m]))[0]
        else:
            lo, hi = max(c - 1, 0), min(c + 1, len(x_pi) - 1)
            while hi > c and not fin[hi]:
                hi -= 1
            if hi > lo:
                out[c] = (np.log(y_t[hi]) - np.log(y_t[lo])) / (np.log(x_pi[hi]) - np.log(x_pi[lo]))
    return out


def first_below(x_pi, s, level):
    """Largest pi (scanning frequent -> rare) at which s first drops below level (log-interp)."""
    for c in range(1, len(s)):
        if np.isfinite(s[c]) and np.isfinite(s[c - 1]) and s[c] < level <= s[c - 1]:
            w = (level - s[c - 1]) / (s[c] - s[c - 1])
            return float(np.exp(np.log(x_pi[c - 1]) + w * (np.log(x_pi[c]) - np.log(x_pi[c - 1])))), c + 1
    return np.nan, None


def first_above(x_pi, s, level):
    for c in range(1, len(s)):
        if np.isfinite(s[c]) and np.isfinite(s[c - 1]) and s[c] >= level > s[c - 1]:
            w = (np.log(level) - np.log(s[c - 1])) / (np.log(s[c]) - np.log(s[c - 1]))
            return float(np.exp(np.log(x_pi[c - 1]) + w * (np.log(x_pi[c]) - np.log(x_pi[c - 1])))), c + 1
    return np.nan, None


def at_pi(x_pi, y, p):
    """log-log interpolation of y(pi) at p (pi decreasing)."""
    m = np.isfinite(y)
    return float(np.exp(np.interp(-np.log(p), -np.log(x_pi[m]), np.log(y[m]),
                                  left=np.nan, right=np.nan)))


def crange(cls):
    cls = np.asarray(cls)
    return f"c{cls.min()}-{cls.max()}" + ("" if len(cls) == cls.max() - cls.min() + 1 else f" (n={len(cls)})")


# --------------------------------------------------------------------------------------
print("# E1b: per-class learning times (timescales.npz: C = 64, alpha = 2, horizon 3000)\n")
print(f"pi_1 = {pi[0]:.3f}, pi_64 = {pi[-1]:.3e}; 1/C^2 = {1 / C**2:.2e}; ell_c(0) = {r['adam_b20.99_b1_e1e-08_loss'][0].mean():.4f} (log 64 = {np.log(64):.4f})")
for k, (b2, b, e) in ADAM.items():
    print(f"  {SHORT[k]:26s} kappa_beta = {kappa_beta(BETA1, b2):.3f}, predicted beta2 kink = {kink_beta2(b2, b):.3e}"
          f" (between classes {np.flatnonzero(pi > kink_beta2(b2, b)).max() + 1} and {np.flatnonzero(pi > kink_beta2(b2, b)).max() + 2}), eps kink b*eps = {b * e:.0e}")
print("PSGD arrays for beta2 = 0.95 and 0.99 identical:",
      np.array_equal(r["psgd_b20.95_b1_e1e-08_loss"], r["psgd_b20.99_b1_e1e-08_loss"]))

# --------------------------------------------------------------------------------------
# Part 1: band slopes and broken power laws
# --------------------------------------------------------------------------------------
BANDS = [("A", 1, 3), ("B", 4, 6), ("C", 7, 14), ("D", 15, 26), ("E", 27, 64)]
METHODS = ["sgd_b1", "sgd_b16", "psgd_b20.99_b1_e1e-08", "psgd_b20.99_b16_e1e-08",
           "psgd_b20.99_b1_e0.0001"] + list(ADAM)
EXPECT = {"sgd_b1": ["-1"] * 5, "sgd_b16": ["-1"] * 5,
          "psgd_b20.99_b1_e1e-08": ["-1/2"] * 4 + ["-1/2 (-1 below 1/C^2)"],
          "psgd_b20.99_b16_e1e-08": ["-1/2"] * 4 + ["-1/2 (-1 below 1/C^2)"],
          "psgd_b20.99_b1_e0.0001": ["-1/2"] * 4 + ["-1/2 (-1 below 1/C^2)"],
          "adam_b20.99_b1_e1e-08": ["-1/2", "crossover", "-1", "-1", "-1"],
          "adam_b20.95_b1_e1e-08": ["-1/2 (c<=3)", "-1", "-1", "-1", "-1"],
          "adam_b20.99_b16_e1e-08": ["-1/2", "-1/2", "-1/2", "-1/2 (kink c~26)", "-1"],
          "adam_b20.99_b1_e0.0001": ["-1/2", "crossover", "-1", "-1", "-1 (eps kink at 1e-4)"]}
print("\n## Part 1. Band slopes d log t_c / d log pi_c (OLS, +- s.e.; n = learned classes in band)\n")
hdr = " | ".join(f"{n}: c{lo}-{hi}, pi {pi[hi - 1]:.1e}-{pi[lo - 1]:.2g}" for n, lo, hi in BANDS)
print(f"| method | {hdr} |")
print("|---" * (len(BANDS) + 1) + "|")
NUM["band_slopes"] = {}
for k in METHODS:
    cells = []
    NUM["band_slopes"][k] = {}
    for (n, lo, hi), ex in zip(BANDS, EXPECT[k]):
        cls = np.arange(lo, hi + 1)
        f = np.isfinite(tc(k)[cls - 1])
        if f.sum() >= 3:
            s, a, se, rms = ols(lpi[cls[f] - 1], np.log(tc(k)[cls[f] - 1]))
            cells.append(f"{s:.3f} +- {se:.3f}" + ("" if f.all() else f" (n={f.sum()})") + f" [exp {ex}]")
            NUM["band_slopes"][k][n] = [float(s), float(se), int(f.sum())]
        else:
            cells.append(f"not learned ({f.sum()}/{len(cls)}) [exp {ex}]")
            NUM["band_slopes"][k][n] = None
    print(f"| {SHORT[k]} | " + " | ".join(cells) + " |")

# single power laws
print("\nSingle power-law fits:")
for k, cls in (("sgd_b1", np.arange(1, 65)), ("sgd_b16", np.arange(1, 65)),
               ("psgd_b20.99_b1_e1e-08", np.arange(1, 9)), ("psgd_b20.99_b16_e1e-08", np.arange(1, 12)),
               ("psgd_b20.99_b16_e1e-08", np.arange(1, 65)), ("adam_b20.99_b1_e0.0001", np.arange(20, 65))):
    s, a, se, rms = ols(lpi[cls - 1], np.log(tc(k)[cls - 1]))
    print(f"  {SHORT[k]:26s} {crange(cls)}: slope {s:.3f} +- {se:.3f}, rms resid {rms:.3f}")
    NUM.setdefault("single_fits", {})[f"{k}:{crange(cls)}"] = [float(s), float(se), float(rms)]

# local slopes
LS = {k: local_slope(pi, tc(k)) for k in METHODS}
NUM["local_slopes"] = {k: [None if not np.isfinite(v) else float(v) for v in LS[k]] for k in METHODS}

print("\n## Part 1. Broken power law t_c = A (pi/pb)^a1 (pi > pb), (pi/pb)^a2 (pi < pb), fit in log-log")
print("free = 4 parameters (break profiled, >= 2 classes per side); fixed = slopes -1/2, -1;"
      " quad = t = A pi^-1/2 sqrt(1 + pb/pi) (theory slopes, smooth). Classes selected by t_c <= cut.\n")
print("| config (pred. kink) | classes | free a1 | free a2 | free pb (x kink) | rms | fixed pb (x kink) | rms | quad pb (x kink) | rms |")
print("|---|---|---|---|---|---|---|---|---|---|")
NUM["bpl"] = {}
for k, (b2, b, e) in ADAM.items():
    kink = kink_beta2(b2, b)
    for cut in (10, 20, 40, np.inf):
        for start in (1, 2):
            sel = np.flatnonzero(np.isfinite(tc(k)) & (tc(k) <= cut)) + 1
            sel = sel[sel >= start]
            if len(sel) < 5:
                continue
            x, y = lpi[sel - 1], np.log(tc(k)[sel - 1])
            xs = np.sort(x)
            f = bpl_sharp(x, y, np.linspace(xs[1] + 1e-9, xs[-2] - 1e-9, 4001))
            g = bpl_sharp(x, y, np.linspace(xs[0], xs[-1], 4001), slopes=(-0.5, -1.0))
            q = bpl_smooth(x, y, np.linspace(xs[0] - 3, xs[-1] + 3, 6001), 2)
            pbf, pbg, pbq = np.exp(f["xb"]), np.exp(g["xb"]), np.exp(q["xb"])
            label = f"t_c<={cut:g}" if np.isfinite(cut) else "all learned"
            print(f"| {SHORT[k]} ({kink:.2e}) | {crange(sel)}, {label} | {f['a1']:.2f} | {f['a2']:.2f} | "
                  f"{pbf:.2e} ({pbf / kink:.2f}) | {f['rms']:.3f} | {pbg:.2e} ({pbg / kink:.2f}) | {g['rms']:.3f} | "
                  f"{pbq:.2e} ({pbq / kink:.2f}) | {q['rms']:.3f} |")
            NUM["bpl"][f"{k}|{label}|start{start}"] = dict(
                classes=crange(sel), a1=float(f["a1"]), a2=float(f["a2"]), pb_free=float(pbf),
                factor_free=float(pbf / kink), rms_free=float(f["rms"]), pb_fixed=float(pbg),
                factor_fixed=float(pbg / kink), rms_fixed=float(g["rms"]), pb_quad=float(pbq),
                factor_quad=float(pbq / kink), rms_quad=float(q["rms"]))
# PSGD: free fits over the whole range (is there a break, and where?)
print("\nPSGD free broken fits (no beta2 kink expected; background kink 1/C^2 = 2.4e-4):")
for k in ("psgd_b20.99_b1_e1e-08", "psgd_b20.99_b16_e1e-08"):
    for hi in (32, 64):
        sel = np.arange(1, hi + 1)
        x, y = lpi[sel - 1], np.log(tc(k)[sel - 1])
        f = bpl_sharp(x, y, np.linspace(x[-2] + 1e-9, x[1] - 1e-9, 4001))
        print(f"  {SHORT[k]:12s} c1-{hi}: a1 {f['a1']:.2f}, a2 {f['a2']:.2f}, pb {np.exp(f['xb']):.2e}"
              f" (= {np.exp(f['xb']) * C**2:.1f} / C^2; {np.exp(f['xb']) / 0.0143:.2f} x the beta2=.99 b=1 kink), rms {f['rms']:.3f}")
    print(f"  {SHORT[k]:12s} local slopes at c=3..8 (pi 0.068 -> 0.0096): {np.round(LS[k][2:8], 3)}")

# local slope crossings
print("\nLocal slope (+-0.2 decade OLS) crossings, scanning from frequent to rare:")
NUM["local_cross"] = {}
for k, (b2, b, e) in ADAM.items():
    kink = kink_beta2(b2, b)
    row = []
    for lev in (-0.75, -1.0, -1.3, -1.5, -2.0):
        p, c = first_below(pi, LS[k], lev)
        row.append(f"{lev}: pi={p:.2e} ({p / kink:.2f}x kink, c~{c}, t_c~{at_pi(pi, tc(k), p):.1f})" if np.isfinite(p) else f"{lev}: never")
        NUM["local_cross"][f"{k}|{lev}"] = [p, (p / kink) if np.isfinite(p) else None]
    print(f"  {SHORT[k]:26s} " + "; ".join(row))

# --------------------------------------------------------------------------------------
# Part 2: classes affected by B growth (Adam b = 1, eps = 1e-8)
# --------------------------------------------------------------------------------------
print("\n## Part 2. Steepening beyond -1 and B_cc growth (Adam b = 1, eps = 1e-8)\n")
k8, k4, k95 = "adam_b20.99_b1_e1e-08", "adam_b20.99_b1_e0.0001", "adam_b20.95_b1_e1e-08"
ratio84 = tc(k8) / tc(k4)
print("Control: eps = 1e-4 (same beta2, b). Frozen rates at Theta_0 (rates.npz, C = 128): eps=1e-4/eps=1e-8 =",
      np.round((np.abs(rates["init_b20.99_b1_e0.0001_adam"]) / np.abs(rates["init_b20.99_b1_e1e-08_adam"]))
               [(rates["priors"] >= 1.5e-4)][[0, 7, 15, 24, 31, 47, -1]], 3),
      "for pi >= 1.5e-4 (classes 1, 8, 16, 25, 32, 48, last); min/max over pi >= 1.5e-4:",
      np.round(np.min((np.abs(rates["init_b20.99_b1_e0.0001_adam"]) / np.abs(rates["init_b20.99_b1_e1e-08_adam"]))[rates["priors"] >= 1.5e-4]), 3),
      np.round(np.max((np.abs(rates["init_b20.99_b1_e0.0001_adam"]) / np.abs(rates["init_b20.99_b1_e1e-08_adam"]))[rates["priors"] >= 1.5e-4]), 3))
print("t_c(eps=1e-8)/t_c(eps=1e-4) for c = 10..30:", np.round(ratio84[9:30], 3))
NUM["ratio_e8_e4"] = [None if not np.isfinite(v) else float(v) for v in ratio84]
for lev in (1.05, 1.1, 1.25, 2.0):
    c = int(np.flatnonzero(np.nan_to_num(ratio84, nan=np.inf) >= lev)[0]) + 1
    print(f"  ratio >= {lev}: from class {c} (pi = {pi[c - 1]:.2e}, t_c = {tc(k8)[c - 1]:.1f} [eps=1e-8], {tc(k4)[c - 1]:.1f} [eps=1e-4])")
    NUM[f"onset_ratio_{lev}"] = [c, float(pi[c - 1]), float(tc(k8)[c - 1])]
d = LS[k8] - LS[k4]
c_dev = int(np.flatnonzero(np.nan_to_num(d, nan=-9) < -0.1)[0]) + 1
print(f"  local slope eps=1e-8 minus eps=1e-4 < -0.1 from class {c_dev} (pi {pi[c_dev - 1]:.2e}, t_c {tc(k8)[c_dev - 1]:.1f}):"
      f" slopes {LS[k8][c_dev - 1]:.2f} vs {LS[k4][c_dev - 1]:.2f}")
s, a, se, rms = ols(lpi[19:64], np.log(tc(k4)[19:64]))
print(f"  eps=1e-4 tail plateau slope c20-64: {s:.3f} +- {se:.3f}; local slope range there {np.nanmin(LS[k4][19:]):.2f}..{np.nanmax(LS[k4][19:]):.2f}")
print(f"  eps=1e-8 learned {np.isfinite(tc(k8)).sum()}/64 (last learned c=29, t_c = {tc(k8)[28]:.0f}); beta2=.95 learned {np.isfinite(tc(k95)).sum()}/64")


def mean_B(k, tt, cls=slice(32, 64)):
    B = r[k + "_Bdiag"][1:, cls].mean(1)
    return float(np.exp(np.interp(np.log(tt), np.log(t[1:]), np.log(B))))


print("\nMean B_cc over the 32 rarest columns (c33-64) [mean over all 64 columns]:")
print("| t | Adam b2=.99 eps=1e-8 | Adam b2=.99 eps=1e-4 | ratio | Adam b2=.95 eps=1e-8 | PSGD b=1 | SGD b=1 | Adam b=16 |")
print("|---|---|---|---|---|---|---|---|")
NUM["meanB_rare"] = {}
for tt in (1, 3, 10, 17, 21, 30, 50, 100, 300, 1000, 3000):
    v = [mean_B(k, tt) for k in (k8, k4, k95, "psgd_b20.99_b1_e1e-08", "sgd_b1", "adam_b20.99_b16_e1e-08")]
    va = [mean_B(k, tt, slice(0, 64)) for k in (k8, k4)]
    print(f"| {tt:g} | {v[0]:.2f} [{va[0]:.2f}] | {v[1]:.2f} [{va[1]:.2f}] | {v[0] / v[1]:.2f} | {v[2]:.2f} | {v[3]:.2f} | {v[4]:.3f} | {v[5]:.2f} |")
    NUM["meanB_rare"][str(tt)] = v


def ito(k, eta=ETA_ADAM):
    s = r[k + "_coef"][:, 2, :]
    cum = np.concatenate([np.zeros((1, C)), np.cumsum(0.5 * (s[1:] + s[:-1]) * np.diff(t)[:, None], 0)])
    return r[k + "_Bdiag"][0] + eta ** 2 * cum


print("\nB_cc / (B_cc(0) + eta^2 int_0^t sigma_cc) averaged over c33-64 (1 = growth fully accounted for by the Ito term),"
      " and growth exponent d log mean(B) / d log t:")
NUM["B_over_ito"] = {}
for k in (k8, k95, "adam_b20.99_b16_e1e-08", k4, "psgd_b20.99_b1_e1e-08"):
    I, B = ito(k), r[k + "_Bdiag"]
    row = []
    for tt in (10, 30, 100, 300, 1000, 3000):
        j = int(np.argmin(np.abs(t - tt)))
        row.append(f"t={t[j]:.0f}: {np.mean(B[j, 32:] / I[j, 32:]):.2f}")
    Bm = B.mean(1)
    ex = []
    for a_, b_ in ((30, 100), (100, 300), (300, 1000), (1000, 3000)):
        ia, ib = int(np.argmin(np.abs(t - a_))), int(np.argmin(np.abs(t - b_)))
        ex.append(f"[{a_},{b_}] {np.log(Bm[ib] / Bm[ia]) / np.log(t[ib] / t[ia]):.2f}")
    print(f"  {SHORT[k]:26s} " + ", ".join(row) + " | exponents " + ", ".join(ex))
    NUM["B_over_ito"][k] = row + ex

print("\nCoefficient diagonals for the rarest class (c = 64; unlearned in eps=1e-8): sigma_cc, chi_cc/pi_c, phi_cc")
for k in (k8, k4):
    co = r[k + "_coef"]
    js = [int(np.argmin(np.abs(t - tt))) for tt in (0.01, 1, 10, 30, 100, 300)]
    print(f"  {SHORT[k]:26s} t={np.round(t[js], 2)}\n      sigma {np.round(co[js, 2, 63], 3)}\n      chi/pi {np.round(co[js, 1, 63] / pi[63], 2)}\n      phi {np.round(co[js, 0, 63], 3)}")
print("\nUnlearned-class loss rise: first t with ell_64(t)/ell_64(0) >= 1.05 / 1.1 / 1.2, and the maximum:")
NUM["bump"] = {}
for k in (k8, k95, k4, "adam_b20.99_b16_e1e-08", "psgd_b20.99_b1_e1e-08", "sgd_b1"):
    L = r[k + "_loss"][:, 63] / r[k + "_loss"][0, 63]
    out = []
    for lev in (1.05, 1.1, 1.2):
        j = np.flatnonzero(L >= lev)
        out.append(float(t[j[0]]) if len(j) else np.nan)
    print(f"  {SHORT[k]:26s} {np.round(out, 1)}  max {L.max():.3f} at t = {t[L.argmax()]:.0f}")
    NUM["bump"][k] = out + [float(L.max())]
mid_ratio = np.abs(rates["mid_b20.99_b1_e0.0001_adam"]) / np.abs(rates["mid_b20.99_b1_e1e-08_adam"])
p128 = rates["priors"]
m = (p128 >= 1.5e-4) & (p128 <= 1e-3)
print(f"\nFrozen rates at the t = 10 Adam state (rates.npz, C = 128): eps=1e-4 / eps=1e-8 rate for pi in [1.5e-4, 1e-3]:"
      f" mean {mid_ratio[m].mean():.2f}, range {mid_ratio[m].min():.2f}-{mid_ratio[m].max():.2f}")
for st in ("init", "mid"):
    for tag in ("b20.99_b1_e1e-08", "b20.99_b1_e0.0001"):
        v = np.abs(rates[f"{st}_{tag}_adam"])
        mm = (p128 >= 1.5e-4) & (p128 < 9e-4)
        print(f"  frozen rate slope ({st}, {tag}) for pi in [1.5e-4, 9e-4]: {ols(np.log(p128[mm]), np.log(v[mm]))[0]:.3f}")
print("Learned classes at t = 3000 (eps=1e-8, b2=.99): ell_c(3000)/ell_c(0) c=20..29:",
      np.round(r[k8 + "_loss"][-1, 19:29] / r[k8 + "_loss"][0, 19:29], 3), " (= running minimum: still decreasing)")
print("PSGD b=1 rare-class loss floor: min_t ell_c/ell_c(0) for c=56..64:",
      np.round((r["psgd_b20.99_b1_e1e-08_loss"] / r["psgd_b20.99_b1_e1e-08_loss"][0]).min(0)[55:], 3))

# --------------------------------------------------------------------------------------
# Part 3: collapse quality
# --------------------------------------------------------------------------------------
print("\n## Part 3. Collapse quality: sd and range (decades) of log10(t_c * scale) over the classes of a regime\n")


def spread(v):
    v = np.log10(v[np.isfinite(v)])
    return float(v.std(ddof=1)), float(v.max() - v.min()), len(v)


def level_times(k, lev):
    L = r[k + "_loss"] / r[k + "_loss"][0]
    return np.array([first_crossing(t, L[:, c], lev) for c in range(C)])


print("| run | regime (classes) | t sqrt(b pi): sd / range | t b pi/sqrt(1-b2) (= t pi up to a constant): sd / range | sd at loss levels 0.9/0.75/0.3, best scaling |")
print("|---|---|---|---|---|")
NUM["collapse"] = {}
for k, b, b2 in (("adam_b20.99_b1_e1e-08", 1, .99), ("adam_b20.99_b16_e1e-08", 16, .99),
                 ("adam_b20.95_b1_e1e-08", 1, .95), ("adam_b20.99_b1_e0.0001", 1, .99)):
    kink = kink_beta2(b2, b)
    T = tc(k)
    W = np.flatnonzero(pi > kink) + 1
    R = np.flatnonzero((pi <= kink) & np.isfinite(T)) + 1
    groups = [("window", W), ("rare, t_c<=20", R[T[R - 1] <= 20]), ("rare, t_c>20", R[T[R - 1] > 20])]
    for gname, cls in groups:
        if len(cls) < 2:
            continue
        s1 = spread(T[cls - 1] * np.sqrt(b * pi[cls - 1]))
        s2 = spread(T[cls - 1] * b * pi[cls - 1] / np.sqrt(1 - b2))
        best = np.sqrt(b * pi) if s1[0] < s2[0] else b * pi
        lv = [spread(level_times(k, lev)[cls - 1] * best[cls - 1])[0] for lev in (0.9, 0.75, 0.3)]
        print(f"| {SHORT[k]} | {gname} ({crange(cls)}) | {s1[0]:.3f} / {s1[1]:.2f} | {s2[0]:.3f} / {s2[1]:.2f} | "
              f"{lv[0]:.3f} / {lv[1]:.3f} / {lv[2]:.3f} ({'sqrt' if s1[0] < s2[0] else 'pi'}) |")
        NUM["collapse"][f"{k}|{gname}"] = dict(classes=crange(cls), sqrt=s1[:2], rare=s2[:2], levels=lv)
for k in ("sgd_b1", "sgd_b16"):
    for lo, hi in ((1, 64), (1, 6), (7, 14), (15, 26), (27, 64)):
        cls = np.arange(lo, hi + 1)
        s1 = spread(tc(k)[cls - 1] * pi[cls - 1])
        s2 = spread(tc(k)[cls - 1] * np.sqrt(pi[cls - 1]))
        lv = [spread(level_times(k, lev)[cls - 1] * pi[cls - 1])[0] for lev in (0.9, 0.75, 0.3)]
        print(f"| {SHORT[k]} | c{lo}-{hi} | {s2[0]:.3f} / {s2[1]:.2f} | t pi: {s1[0]:.3f} / {s1[1]:.2f} | {lv[0]:.3f} / {lv[1]:.3f} / {lv[2]:.3f} (pi) |")
        NUM["collapse"][f"{k}|c{lo}-{hi}"] = dict(pi=s1[:2], sqrt=s2[:2], levels=lv)

print("\nCross-configuration tests of the scalings:")
wb = tc("adam_b20.99_b1_e1e-08") / (4 * tc("adam_b20.99_b16_e1e-08"))
wp = tc("psgd_b20.99_b1_e1e-08") / (4 * tc("psgd_b20.99_b16_e1e-08"))
print(f"  window sqrt(b): t_c(b=1)/(sqrt(16) t_c(b=16)) Adam c1-8 {np.round(wb[:8], 3)}; PSGD c1-8 {np.round(wp[:8], 3)}, PSGD c26 {wp[25]:.2f}, c64 {wp[63]:.2f}")
r95 = tc("adam_b20.95_b1_e1e-08") / tc("adam_b20.99_b1_e1e-08")
pre = lambda b2: 1 / (np.sqrt(1 - b2) * (1 - BETA1 / np.sqrt(b2)))
print(f"  beta2: t_c(.95)/t_c(.99) c1-14 {np.round(r95[:14], 3)}; predicted 1 (window), {np.sqrt(5):.2f} (plan's b pi/sqrt(1-b2)),"
      f" {pre(.99) / pre(.95):.2f} (rare-event law with its (1 - b1/sqrt(b2)) factor)")
NUM["beta2_ratio"] = [float(v) for v in r95[:14]]
for k, b, b2 in (("adam_b20.99_b1_e1e-08", 1, .99), ("adam_b20.95_b1_e1e-08", 1, .95),
                 ("adam_b20.99_b16_e1e-08", 16, .99), ("adam_b20.99_b1_e0.0001", 1, .99)):
    kink = kink_beta2(b2, b)
    T = tc(k)
    cls = np.flatnonzero((pi <= kink) & (T <= 20)) + 1
    v = T[cls - 1] * b * pi[cls - 1] / np.sqrt(1 - b2)
    vfull = T[cls - 1] * b * pi[cls - 1] * pre(b2) / pre(.99) / np.sqrt(1 - .99)
    print(f"  rare-event collapse value t_c b pi/sqrt(1-b2), {SHORT[k]} ({crange(cls)}, pi/kink {pi[cls[-1] - 1] / kink:.2f}-{pi[cls[0] - 1] / kink:.2f}):"
          f" median {np.median(v):.3f} (range {v.min():.3f}-{v.max():.3f}); with the (1-b1/sqrt(b2)) factor (normalised to b2=.99): {np.median(vfull):.3f}")
    NUM.setdefault("rare_value", {})[k] = [float(np.median(v)), float(np.median(vfull))]
print("  u-collapse F = t_c sqrt(b pi) at fixed u = pi / kink (log-log interpolation):")
NUM["F_at_u"] = {}
for u in (0.3, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0):
    vals = {}
    for k, (b2, b, e) in ADAM.items():
        kink = kink_beta2(b2, b)
        F = tc(k) * np.sqrt(b * pi)
        vals[k] = at_pi(pi, F, u * kink)
    fin = [v for v in vals.values() if np.isfinite(v)]
    print(f"    u={u:g}: " + ", ".join(f"{SHORT[k]} {v:.3f}" for k, v in vals.items() if np.isfinite(v))
          + (f"  -> max/min {max(fin) / min(fin):.2f}" if len(fin) > 1 else ""))
    NUM["F_at_u"][str(u)] = {k: (v if np.isfinite(v) else None) for k, v in vals.items()}

# --------------------------------------------------------------------------------------
# Part 4: ODE vs simulation (Adam b2=.99, b=1, eps=1e-8; D = 4000, 8 runs, t <= 300)
# --------------------------------------------------------------------------------------
print("\n## Part 4. ODE vs simulation (8 runs, D = 4000, t <= 300)\n")
ct, cl, ctc = r["check_time"], r["check_loss"], r["check_tc"]
nr = cl.shape[0]
ode = tc(k8)
Lc = interp_log(ct, t, r[k8 + "_loss"])
ode_grid = np.array([first_crossing(ct, Lc[:, c], Lc[0, c] / 2) for c in range(C)])
print(f"ODE t_c recomputed on the simulation grid differs from the stored one by at most "
      f"{np.nanmax(np.abs(ode_grid[:27] / ode[:27] - 1)):.1e} (classes 1-27)")
print("| c | pi | ODE t_c | sim median | sim min-max | runs learned | median/ODE - 1 | mean log(sim/ODE) +- s.e. | z |")
print("|---|---|---|---|---|---|---|---|---|")
NUM["sim"] = {}
zs = []
for c in range(29):
    x = ctc[:, c]
    f = np.isfinite(x)
    if f.sum() == 0:
        print(f"| {c + 1} | {pi[c]:.2e} | {ode[c]:.1f} | - | - | 0/{nr} | - | - | - |")
        continue
    med = np.median(np.where(f, x, np.inf))
    lr = np.log(x[f] / ode[c])
    se = lr.std(ddof=1) / np.sqrt(f.sum()) if f.sum() > 1 else np.nan
    z = lr.mean() / se if f.all() else np.nan
    if f.all():
        zs.append((c + 1, z, lr.mean(), se))
    print(f"| {c + 1} | {pi[c]:.2e} | {ode[c]:.3g} | {med:.3g} | {x[f].min():.3g}-{x[f].max():.3g} | {f.sum()}/{nr} | "
          f"{med / ode[c] - 1:+.3f} | {lr.mean():+.3f} +- {se:.3f} | {z:+.1f} |")
    NUM["sim"][c + 1] = dict(ode=float(ode[c]), median=float(med), n=int(f.sum()), mean_log=float(lr.mean()), se=float(se))
zs = np.array(zs)
for lo, hi in ((1, 4), (5, 27)):
    m = (zs[:, 0] >= lo) & (zs[:, 0] <= hi)
    chi2 = float(np.sum(zs[m, 1] ** 2))
    w = 1 / zs[m, 3] ** 2
    pooled, pse = float(np.sum(w * zs[m, 2]) / w.sum()), float(1 / np.sqrt(w.sum()))
    print(f"  classes {lo}-{hi}: sum z^2 = {chi2:.1f} on {m.sum()} classes; inverse-variance pooled mean log ratio {pooled:+.4f} +- {pse:.4f};"
          f" max |z| {np.abs(zs[m, 1]).max():.1f}; per-run sd of log t_c median {np.median(zs[m, 3] * np.sqrt(nr)):.3f}")
    NUM[f"sim_pooled_{lo}_{hi}"] = [chi2, int(m.sum()), pooled, pse]
# class 28 censoring test
sd_run = np.median(zs[(zs[:, 0] >= 22) & (zs[:, 0] <= 27), 3] * np.sqrt(nr))
p_learn = 0.5 * (1 + erf(np.log(300 / ode[27]) / sd_run / sqrt(2)))
n28 = int(np.isfinite(ctc[:, 27]).sum())
pval = sum(comb(nr, j) * p_learn ** j * (1 - p_learn) ** (nr - j) for j in range(n28, nr + 1))
print(f"  class 28 (ODE t_c = {ode[27]:.0f} > 300): {n28}/{nr} runs crossed by t = 300; with per-run log-sd {sd_run:.2f}"
      f" (classes 22-27) the ODE predicts P(cross) = {p_learn:.2f}, i.e. {nr * p_learn:.1f} runs; P(>= {n28}) = {pval:.2f}")
print(f"  class 29 (ODE t_c = {ode[28]:.0f}): {int(np.isfinite(ctc[:, 28]).sum())}/{nr} runs crossed by t = 300")
NUM["class28"] = [n28, float(p_learn), float(pval)]
print("\nPer-class losses, ODE (interpolated in log t) vs simulation mean over 8 runs:")
print("| t | median abs rel diff | max abs rel diff (class) | median abs z | max abs z (class) | classes with abs z > 3 |")
print("|---|---|---|---|---|---|")
NUM["sim_loss"] = {}
for tt in (0.1, 1, 10, 30, 100, 300):
    j = int(np.argmin(np.abs(ct - tt)))
    sm, se = cl[:, j].mean(0), cl[:, j].std(0, ddof=1) / np.sqrt(nr)
    rel, z = (Lc[j] - sm) / sm, (Lc[j] - sm) / se
    big = [int(v) for v in np.flatnonzero(np.abs(z) > 3) + 1]
    print(f"| {ct[j]:.3g} | {np.median(np.abs(rel)):.4f} | {np.abs(rel).max():.3f} ({np.abs(rel).argmax() + 1}) | "
          f"{np.median(np.abs(z)):.2f} | {np.abs(z).max():.1f} ({np.abs(z).argmax() + 1}) | {list(big)} |")
    NUM["sim_loss"][str(tt)] = [float(np.median(np.abs(rel))), float(np.abs(rel).max()), float(np.median(np.abs(z))), int(len(big))]
R_ode, R_sim = (Lc * pi).sum(1), (cl * pi).sum(2)
gap, gse = R_sim.mean(0) - R_ode, R_sim.std(0, ddof=1) / np.sqrt(nr)
m = ct >= 1
print(f"Total risk: t>=1 max |gap| {np.abs(gap[m]).max():.4f} = {np.max(np.abs(gap[m]) / R_ode[m]):.4f} of the risk, max |z| {np.max(np.abs(gap[m]) / gse[m]):.1f};"
      f" t~0.1 gap {gap[np.argmin(np.abs(ct - 0.1))]:+.3f} ({gap[np.argmin(np.abs(ct - 0.1))] / R_ode[np.argmin(np.abs(ct - 0.1))]:+.3f} rel,"
      f" z {gap[np.argmin(np.abs(ct - 0.1))] / gse[np.argmin(np.abs(ct - 0.1))]:.0f})")
NUM["risk_gap_t_ge_1"] = [float(np.abs(gap[m]).max()), float(np.max(np.abs(gap[m]) / R_ode[m])), float(np.max(np.abs(gap[m]) / gse[m]))]

# --------------------------------------------------------------------------------------
# Part 5: Adam / PSGD lag
# --------------------------------------------------------------------------------------
print("\n## Part 5. Lag ratio t_c(Adam)/t_c(PSGD) vs pi_c\n")


def frozen_ratio(tag):
    q = np.abs(rates[f"init_{tag}_psgd"]) / np.abs(rates[f"init_{tag}_adam"])
    return np.exp(np.interp(-lpi, -np.log(p128), np.log(q)))


print("| config (kink) | ratio at c=1 | first reaches 1.1 / 1.25 / 1.5 / 2 at pi (x kink) | frozen ratio (Theta_0, C=128) reaches 1.25 / 1.5 / 2 at (x kink) | ratio at u=1 (frozen) | free broken fit of log ratio (t_c<=20): slopes, break (x kink) | fixed (0,-1/2): break (x kink) |")
print("|---|---|---|---|---|---|---|")
NUM["lag"] = {}
for tag, b2, b in (("b20.99_b1_e1e-08", .99, 1), ("b20.95_b1_e1e-08", .95, 1),
                   ("b20.99_b16_e1e-08", .99, 16), ("b20.99_b1_e0.0001", .99, 1)):
    ta, tp = tc("adam_" + tag), tc("psgd_" + tag)
    ratio, kink, fr = ta / tp, kink_beta2(b2, b), frozen_ratio(tag)
    ons = [first_above(pi, ratio, lev)[0] for lev in (1.1, 1.25, 1.5, 2.0)]
    fons = [first_above(pi, fr, lev)[0] for lev in (1.25, 1.5, 2.0)]
    msk = np.isfinite(ratio) & (ta <= 20)
    x, y = lpi[msk], np.log(ratio[msk])
    xs = np.sort(x)
    f = bpl_sharp(x, y, np.linspace(xs[1] + 1e-9, xs[-2] - 1e-9, 4001))
    g = bpl_sharp(x, y, np.linspace(xs[0], xs[-1], 4001), slopes=(0.0, -0.5))
    print(f"| Adam {tag} ({kink:.2e}) | {ratio[0]:.2f} | " + " / ".join(f"{o:.2e} ({o / kink:.2f})" for o in ons)
          + " | " + " / ".join(f"{o / kink:.2f}" for o in fons)
          + f" | {at_pi(pi, ratio, kink):.2f} ({at_pi(pi, fr, kink):.2f}) | {f['a1']:+.2f}/{f['a2']:+.2f}, {np.exp(f['xb']) / kink:.2f} ({crange(np.flatnonzero(msk) + 1)}) | {np.exp(g['xb']) / kink:.2f} |")
    NUM["lag"][tag] = dict(ratio_c1=float(ratio[0]), onsets=[o / kink for o in ons], frozen_onsets=[o / kink for o in fons],
                           ratio_at_kink=at_pi(pi, ratio, kink), frozen_at_kink=at_pi(pi, fr, kink),
                           free=[float(f["a1"]), float(f["a2"]), float(np.exp(f["xb"]) / kink)],
                           fixed=float(np.exp(g["xb"]) / kink))
print("Theory: ratio = 1 above the kink and (kink/pi)^(1/2) below; sharp: 1 at the kink, 'sum' model 2, quadrature 1.41.")
print("Error-based t_c (err_c <= 1/2) anchor slopes are not power laws even for SGD (c1-2 -1.06, c2-3 -1.53, c3-4 -1.84, c4-6 -1.68): not used.")

with open(SCR + "/e1b_numbers.json", "w") as fh:
    json.dump(NUM, fh, indent=1, default=lambda o: None if (isinstance(o, float) and not np.isfinite(o)) else (o.tolist() if hasattr(o, "tolist") else str(o)))
print("\nwrote", SCR + "/e1b_numbers.json")
