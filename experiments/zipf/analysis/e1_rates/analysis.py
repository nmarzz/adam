"""E1a frozen per-class rates (H1): slopes, kinks, PSGD comparison, simulation check,
precision and psi diagnostics, from figs/experiments/zipf/rates.npz and the
forcing/restoring decomposition (decompose.py -> decomp_init.npz, decomp_mid.npz).

Run: cd <repo> && OMP_NUM_THREADS=1 JAX_PLATFORMS=cpu real/bin/python analysis.py
Writes results.json next to this file and prints the tables."""
import json
import sys
import numpy as np
from scipy.optimize import least_squares

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from common import *

r = load()
pi = r["priors"]
C = len(pi)
x_all = np.log(pi)
dec = {s: dict(np.load(f"{OUTDIR}/decomp_{s}.npz")) for s in ("init", "mid")}
R2 = MEAN_NORM ** 2
OUT = {}

# decade bands in pi (class ranges, 1-based)
BANDS = [(pi.min() * 0.999, 1e-4), (1e-4, 1e-3), (1e-3, 1e-2), (1e-2, 1.0)]


def classes_in(lo, hi):
    c = np.flatnonzero((pi >= lo) & (pi < hi))
    return c


def ols_slope(c, y):
    """OLS slope of log y on log pi over classes c, with its regression s.e."""
    X, Y = x_all[c], np.log(y[c])
    A = np.vstack([X, np.ones_like(X)]).T
    coef, res, *_ = np.linalg.lstsq(A, Y, rcond=None)
    if len(c) > 2:
        s2 = np.sum((Y - A @ coef) ** 2) / (len(c) - 2)
        se = np.sqrt(s2 / np.sum((X - X.mean()) ** 2))
    else:
        se = np.nan
    return coef[0], se


def local_slope(y, h=0.12):
    """Gaussian-kernel local linear regression of log y on log10 pi (bandwidth h
    decades, widened to include >= 3 classes)."""
    lx, ly = np.log10(pi), np.log(y)
    out = np.empty(C)
    for i in range(C):
        d = np.abs(lx - lx[i])
        hh = max(h, np.sort(d)[2])
        w = np.exp(-0.5 * (d / hh) ** 2)
        W = w.sum()
        mx, my = (w * lx).sum() / W, (w * ly).sum() / W
        out[i] = (w * (lx - mx) * (ly - my)).sum() / (w * (lx - mx) ** 2).sum() / np.log(10)
    return out


def sbpl(p, x):
    """Smoothly broken power law in x = ln pi: slope s1 below, s2 above ln pi_b;
    the asymptotes intersect exactly at pi_b."""
    lnA, s1, s2, xb, lnD = p
    D = np.exp(lnD)
    z = (x - xb) / D
    soft = np.logaddexp(0.0, z) - np.log(2.0)        # log((1 + e^z)/2)
    return lnA + s1 * (x - xb) + (s2 - s1) * D * soft


def fit_sbpl(c, y, fixed=None, xb0=None):
    """Least squares in log space. fixed=(s1, s2) fixes the slopes."""
    X, Y = x_all[c], np.log(y[c])
    if xb0 is None:
        xb0 = np.median(X)
    best = None
    for xb_start in (xb0, xb0 - 1.5, xb0 + 1.5):
        for lnD0 in (np.log(0.3), np.log(1.0)):
            if fixed is None:
                p0 = [np.interp(xb_start, X[::-1], Y[::-1]), 1.0, 0.5, xb_start, lnD0]
                lb = [-50, -0.5, -0.5, x_all.min() - 5, np.log(0.05)]
                ub = [50, 2.5, 2.5, x_all.max() + 5, np.log(5.0)]
                fun = lambda p: sbpl(p, X) - Y
            else:
                s1, s2 = fixed
                p0 = [np.interp(xb_start, X[::-1], Y[::-1]), xb_start, lnD0]
                lb = [-50, x_all.min() - 5, np.log(0.05)]
                ub = [50, x_all.max() + 5, np.log(5.0)]
                fun = lambda q: sbpl([q[0], s1, s2, q[1], q[2]], X) - Y
            p0 = np.clip(p0, np.array(lb) + 1e-9, np.array(ub) - 1e-9)
            sol = least_squares(fun, p0, bounds=(lb, ub))
            if best is None or sol.cost < best.cost:
                best = sol
    p = best.x if fixed is None else np.array([best.x[0], fixed[0], fixed[1], best.x[1], best.x[2]])
    rms = np.sqrt(2 * best.cost / len(c))
    return {"s1": p[1], "s2": p[2], "pi_b": float(np.exp(p[3])), "delta": float(np.exp(p[4])),
            "rms_log": float(rms), "p": p}


def slope_crossing(s, level=0.75, lo=None, hi=None):
    """pi where the local slope crosses `level` (log interpolation), scanning from
    the frequent end toward the rare end; None if no crossing."""
    order = np.argsort(pi)          # rare -> frequent
    ps, ss = pi[order], s[order]
    m = np.ones_like(ps, bool)
    if lo is not None:
        m &= ps >= lo
    if hi is not None:
        m &= ps <= hi
    ps, ss = ps[m], ss[m]
    for i in range(len(ps) - 1, 0, -1):
        a, b = ss[i] - level, ss[i - 1] - level
        if a == 0:
            return float(ps[i])
        if a * b < 0:
            w = a / (a - b)
            return float(np.exp(np.log(ps[i]) + w * (np.log(ps[i - 1]) - np.log(ps[i]))))
    return None


def fmt_cls(c):
    return f"{c.min() + 1}-{c.max() + 1}" if len(c) else "-"


# --------------------------------------------------------------------------- #
# 1. band slopes, local slopes, kink fits
# --------------------------------------------------------------------------- #
print("=" * 100)
print("1. Band slopes of |rate| vs pi (OLS in log-log, +- regression s.e.); bands are decades in pi")
print("   classes per band:", [fmt_cls(classes_in(*b)) for b in BANDS])
OUT["bands"] = {f"{lo:.1e}-{hi:.0e}": fmt_cls(classes_in(lo, hi)) for lo, hi in BANDS}
OUT["slopes"] = {}
OUT["kinks"] = {}
OUT["local_slopes"] = {}
for state in ("init", "mid"):
    print(f"\n--- state {state} ---")
    print(f"{'config':22s} {'series':8s}  " + "  ".join(f"[{lo:.0e},{hi:.0e})".rjust(15) for lo, hi in BANDS))
    for b2, b, e in CONFIGS:
        t = tag(b2, b, e)
        series = {"total": np.abs(r[f"{state}_{t}_adam"]),
                  "forcing": dec[state][f"{t}_forcing"],
                  "psgd": np.abs(r[f"{state}_{t}_psgd"])}
        for name, y in series.items():
            if name == "psgd" and (b2, e) != (0.99, 1e-8) and state == "init" and b == 1 and b2 != 0.99:
                pass
            row = []
            for lo, hi in BANDS:
                c = classes_in(lo, hi)
                s, se = ols_slope(c, y)
                row.append((s, se))
                OUT["slopes"][f"{state}/{t}/{name}/{lo:.0e}"] = [float(s), float(se)]
            print(f"{t:22s} {name:8s}  " + "  ".join(f"{s:6.3f}+-{se:5.3f}".rjust(15) for s, se in row))
        OUT["local_slopes"][f"{state}/{t}/forcing"] = local_slope(series["forcing"]).tolist()
        OUT["local_slopes"][f"{state}/{t}/total"] = local_slope(series["total"]).tolist()
        OUT["local_slopes"][f"{state}/{t}/psgd"] = local_slope(series["psgd"]).tolist()
    sg = np.abs(r[f"{state}_b20.99_b1_e1e-08_sgd"])
    row = [ols_slope(classes_in(lo, hi), sg) for lo, hi in BANDS]
    print(f"{'(any)':22s} {'sgd':8s}  " + "  ".join(f"{s:6.3f}+-{se:5.3f}".rjust(15) for s, se in row))

# ---- kinks at Theta_0 --------------------------------------------------------
print("\n" + "=" * 100)
print("2. Kinks at Theta_0. Fits on the forcing part (D -> inf rate at O = 0, all 128 classes)")
print("   and on the stored |rate| for classes 1-64 (tail contaminated by -(phi^T O)_cc).")
print("   sbpl-free: smooth broken power law, slopes/break/smoothness free (pi_b = asymptote intersection)")
print("   sbpl-fix : slopes fixed to 1 (rare side) and 1/2 (frequent side)")
print("   x3/4     : pi where the local slope crosses 3/4")
print("   pref     : intersection of r^2 sqrt(b pi) with F x (rare-event or eps law), F = median ratio in")
print("              the deepest decade (classes 79-128) of the forcing part")
hdr = (f"{'config':20s} {'pred b2-kink':>12s} {'pred eps':>9s} | {'free s1':>7s} {'s2':>6s} {'pi_b':>9s} "
       f"{'rms':>5s} | {'fix pi_b':>9s} {'rms':>5s} | {'x3/4':>9s} | {'F_rare':>6s} {'F_eps':>6s} "
       f"{'pref pi_b':>9s} | free(tot,1-64) s1 s2 pi_b")
print(hdr)
deep = classes_in(*BANDS[0])
for b2, b, e in CONFIGS:
    t = tag(b2, b, e)
    p = predicted_rates(pi, BETA1, b2, e, b)
    yF = dec["init"][f"{t}_forcing"]
    yT = np.abs(r[f"init_{t}_adam"])
    allc = np.arange(C)
    head = np.arange(64)
    k_pred = p["kink_beta2"]
    xb0 = np.log(max(k_pred, b * e, pi.min() * 2))
    free = fit_sbpl(allc, yF, xb0=xb0)
    fix = fit_sbpl(allc, yF, fixed=(1.0, 0.5), xb0=xb0)
    freeT = fit_sbpl(head, yT, xb0=xb0)
    s_loc = np.array(OUT["local_slopes"][f"init/{t}/forcing"])
    x34 = slope_crossing(s_loc, 0.75)
    F_rare = float(np.median(yF[deep] / p["rare"][deep]))
    eps_law = R2 * pi * (1 - 1 / C) / np.sqrt(e)
    F_eps = float(np.median(yF[deep] / eps_law[deep]))
    # slope-1 asymptote a*pi (a = median of yF/pi in the deepest decade) meets r^2 sqrt(b pi)
    a = float(np.median(yF[deep] / pi[deep]))
    pref = (R2 * np.sqrt(b) / a) ** 2
    OUT["kinks"][t] = dict(pred_beta2=float(k_pred), pred_eps=float(b * e), kappa=float(kappa_beta(BETA1, b2)),
                           free=dict(s1=float(free["s1"]), s2=float(free["s2"]), pi_b=free["pi_b"],
                                     delta=free["delta"], rms=free["rms_log"]),
                           fixed=dict(pi_b=fix["pi_b"], delta=fix["delta"], rms=fix["rms_log"]),
                           free_total_1_64=dict(s1=float(freeT["s1"]), s2=float(freeT["s2"]),
                                                pi_b=freeT["pi_b"], rms=freeT["rms_log"]),
                           x34=x34, F_rare=F_rare, F_eps=F_eps, pref=pref)
    print(f"{t:20s} {k_pred:12.3e} {b * e:9.1e} | {free['s1']:7.3f} {free['s2']:6.3f} {free['pi_b']:9.2e} "
          f"{free['rms_log']:5.3f} | {fix['pi_b']:9.2e} {fix['rms_log']:5.3f} | "
          f"{(x34 if x34 else float('nan')):9.2e} | {F_rare:6.3f} {F_eps:6.3f} {pref:9.2e} | "
          f"{freeT['s1']:.3f} {freeT['s2']:.3f} {freeT['pi_b']:.2e}")

print("\n   measured / predicted beta2-kink (b2 sweep and b sweep, eps = 1e-8):")
print(f"   {'config':20s} {'n_b=b pi_b/(1-b2)':>18s} {'kappa':>6s} {'free/pred':>9s} {'fix/pred':>9s} {'x3/4/pred':>9s} {'pref/pred':>9s} {'1/F_rare^2':>10s}")
for b2, b, e in CONFIGS[:6]:
    t = tag(b2, b, e)
    k = OUT["kinks"][t]
    pr = k["pred_beta2"]
    print(f"   {t:20s} {b * k['fixed']['pi_b'] / (1 - b2):18.3f} {k['kappa']:6.3f} {k['free']['pi_b'] / pr:9.2f} "
          f"{k['fixed']['pi_b'] / pr:9.2f} {(k['x34'] or np.nan) / pr:9.2f} {k['pref'] / pr:9.2f} {1 / k['F_rare'] ** 2:10.2f}")
print("\n   eps sweep: measured kink vs predicted pi = b eps and vs the beta2 kink:")
print(f"   {'config':20s} {'b*=sqrt((1-b2)/eps)':>20s} {'fix pi_b':>9s} {'/(b eps)':>9s} {'/b2-kink':>9s} {'x3/4':>9s} {'/(b eps)':>9s}")
for b2, b, e in CONFIGS[6:]:
    t = tag(b2, b, e)
    k = OUT["kinks"][t]
    x34 = k["x34"] or np.nan
    print(f"   {t:20s} {np.sqrt((1 - b2) / e):20.2f} {k['fixed']['pi_b']:9.2e} {k['fixed']['pi_b'] / (b * e):9.2f} "
          f"{k['fixed']['pi_b'] / k['pred_beta2']:9.2f} {x34:9.2e} {x34 / (b * e):9.2f}")

# PSGD background kink
print("\n   PSGD at Theta_0: pi where the local slope crosses 3/4 (expected background kink "
      f"pi(1-1/C)^2 = (1-pi)/C^2 -> {1 / (C - 1) ** 2:.2e} for eps << 1/C^2/b):")
for b2, b, e in CONFIGS:
    t = tag(b2, b, e)
    s_ps = np.array(OUT["local_slopes"][f"init/{t}/psgd"])
    xp = slope_crossing(s_ps, 0.75)
    OUT["kinks"][t]["psgd_x34"] = xp
    # where the predicted kink of PSGD with background and eps: E f^2 / b == ... solve numerically
    print(f"   {t:20s} psgd x3/4 = {xp if xp is None else f'{xp:.2e}'}  (pred eps kink b eps = {b * e:.1e})")

# ---- Adam / PSGD and the collapse in n = b pi / (1 - beta2) ------------------
print("\n" + "=" * 100)
print("3. Adam vs PSGD and vs the rare-event law at Theta_0 (forcing part; total in brackets for c <= 64)")
OUT["collapse"] = {}
for b2, b, e in CONFIGS:
    t = tag(b2, b, e)
    p = predicted_rates(pi, BETA1, b2, e, b)
    yF = dec["init"][f"{t}_forcing"]
    ps = np.abs(r[f"init_{t}_psgd"])
    W = R2 * np.sqrt(b * pi)
    n = b * pi / (1 - b2)
    OUT["collapse"][t] = dict(n=n.tolist(), adam_over_W=(yF / W).tolist(), adam_over_psgd=(yF / ps).tolist(),
                              adam_over_rare=(yF / p["rare"]).tolist(), psgd_over_W=(ps / W).tolist())
    sel = [0, 1, 2, 4, 7, 15, 24, 31, 49, 63, 77, 99, 127]
    print(f"  {t}: class   " + " ".join(f"{c + 1:7d}" for c in sel))
    print(f"  {'':18s} n        " + " ".join(f"{n[c]:7.3g}" for c in sel))
    print(f"  {'':18s} Adam/PSGD" + " ".join(f"{yF[c] / ps[c]:7.3f}" for c in sel))
    print(f"  {'':18s} Adam/rare" + " ".join(f"{yF[c] / p['rare'][c]:7.3f}" for c in sel))
    print(f"  {'':18s} Adam/W   " + " ".join(f"{yF[c] / W[c]:7.3f}" for c in sel))

# where Adam/PSGD crosses 1/2 and 0.9 in n (collapse test)
print("\n   collapse test (eps = 1e-8 configs): n at which Adam/PSGD (forcing) = 0.5, 0.75, 0.9")
OUT["collapse_n"] = {}
for b2, b, e in CONFIGS[:6]:
    t = tag(b2, b, e)
    n = np.array(OUT["collapse"][t]["n"])
    ratio = np.array(OUT["collapse"][t]["adam_over_psgd"])
    order = np.argsort(n)
    res = []
    for lev in (0.5, 0.75, 0.9):
        nn, rr = n[order], ratio[order]
        # monotone part: use last crossing from below
        idx = np.flatnonzero((rr[:-1] < lev) & (rr[1:] >= lev))
        if len(idx):
            i = idx[-1]
            w = (lev - rr[i]) / (rr[i + 1] - rr[i])
            res.append(float(np.exp(np.log(nn[i]) + w * (np.log(nn[i + 1]) - np.log(nn[i])))))
        else:
            res.append(None)
    OUT["collapse_n"][t] = res
    kap = kappa_beta(BETA1, b2)
    print(f"   {t:20s} n(0.5)={res[0] if res[0] is None else f'{res[0]:.3f}'}  n(0.75)="
          f"{res[1] if res[1] is None else f'{res[1]:.3f}'}  n(0.9)={res[2] if res[2] is None else f'{res[2]:.3f}'}"
          f"   n(0.5)/kappa={(res[0] / kap) if res[0] else float('nan'):.3f}")

# --------------------------------------------------------------------------- #
# 4. simulation check
# --------------------------------------------------------------------------- #
print("\n" + "=" * 100)
print("4. Direct frozen-parameter simulation vs ODE coefficients (b20.99_b1_e1e-08 at Theta_0)")
t = "b20.99_b1_e1e-08"
cc = r["check_classes"]
ode, ode_se = r[f"init_{t}_adam"][cc], r[f"init_{t}_adam_se"][cc]
sim, sim_se = r["check_rate"], r["check_se"]
z = (sim - ode) / np.sqrt(ode_se ** 2 + sim_se ** 2)
OUT["sim_check"] = []
for i, c in enumerate(cc):
    row = dict(cls=int(c + 1), pi=float(pi[c]), ode=float(ode[i]), ode_se=float(ode_se[i]), sim=float(sim[i]),
               sim_se=float(sim_se[i]), rel=float(sim[i] / ode[i] - 1), z=float(z[i]),
               forcing=float(dec["init"][f"{t}_forcing"][c]),
               restore=float(r[f"init_{t}_adam"][c] - dec["init"][f"{t}_forcing"][c]))
    OUT["sim_check"].append(row)
    print(f"   class {c + 1:3d} (pi={pi[c]:.3e}): ODE {ode[i]:.5f} +- {ode_se[i]:.5f}  sim {sim[i]:.5f} +- {sim_se[i]:.5f}"
          f"  rel diff {100 * (sim[i] / ode[i] - 1):+.2f}%  z = {z[i]:+.2f}   (restoring part {row['restore']:+.5f})")

# --------------------------------------------------------------------------- #
# 5. precision
# --------------------------------------------------------------------------- #
print("\n" + "=" * 100)
print("5. Seed-to-seed relative s.e. of the Adam rate (4 seeds)")
OUT["precision"] = {}
for state in ("init", "mid"):
    for b2, b, e in CONFIGS:
        t = tag(b2, b, e)
        a, se = r[f"{state}_{t}_adam"], r[f"{state}_{t}_adam_se"]
        F = dec[state][f"{t}_forcing"]
        rel = se / np.abs(a)
        relF = se / F
        big = np.flatnonzero(rel > 0.02)
        OUT["precision"][f"{state}/{t}"] = dict(max_rel=float(rel.max()), argmax=int(rel.argmax() + 1),
                                               max_se_over_forcing=float(relF.max()),
                                               n_rel_gt_2pct=int((rel > 0.02).sum()),
                                               n_rel_gt_5pct=int((rel > 0.05).sum()),
                                               classes_gt_2pct=(big + 1).tolist(),
                                               median_rel=float(np.median(rel)))
        print(f"   {state:4s} {t:20s} max rel s.e. {rel.max():.3f} (class {rel.argmax() + 1}, rate {a[rel.argmax()]:+.2e},"
              f" forcing {F[rel.argmax()]:.2e}); max s.e./forcing {relF.max():.4f} (class {relF.argmax() + 1});"
              f" median rel {np.median(rel):.4f}; #>2%: {len(big)} {list(big + 1) if len(big) < 12 else ''}")

# --------------------------------------------------------------------------- #
# 6. mid state
# --------------------------------------------------------------------------- #
print("\n" + "=" * 100)
print("6. Mid state (Adam ODE at t = 10): per-class loss, O_cc, B_cc; rates vs Theta_0")
ml, il = r["mid_loss"], r["init_loss"]
learned = np.flatnonzero(ml <= il / 2)
print(f"   classes with ell_c(10) <= ell_c(0)/2: {fmt_cls(learned)} ({len(learned)}); ell_c(10) > ell_c(0): "
      f"{fmt_cls(np.flatnonzero(ml > il))}")
OUT["mid"] = dict(learned=(learned + 1).tolist(), peak={}, ratio_mid_init={}, restore_frac={})
for b2, b, e in CONFIGS:
    t = tag(b2, b, e)
    am, ai = np.abs(r[f"mid_{t}_adam"]), dec["init"][f"{t}_forcing"]
    Fm = dec["mid"][f"{t}_forcing"]
    peak = int(np.argmax(am))
    rat = am / ai
    resto = (r[f"mid_{t}_adam"] - Fm) / Fm
    OUT["mid"]["peak"][t] = [peak + 1, float(pi[peak])]
    OUT["mid"]["ratio_mid_init"][t] = rat.tolist()
    OUT["mid"]["restore_frac"][t] = resto.tolist()
    sel = [0, 2, 7, 15, 24, 49, 99, 127]
    print(f"   {t:20s} peak at class {peak + 1} (pi={pi[peak]:.2e}); mid/init: " +
          " ".join(f"c{c + 1}:{rat[c]:.2f}" for c in sel) + " | restoring/forcing at mid: " +
          " ".join(f"c{c + 1}:{resto[c]:+.2f}" for c in sel))
    print(f"   {'':20s} mid forcing / init forcing: " + " ".join(f"c{c + 1}:{Fm[c] / ai[c]:.3f}" for c in sel))
# mid kinks: fit on unlearned classes only
print("\n   mid-state fits on classes with ell_c(10) > ell_c(0)/2 (unlearned): SBPL free and fixed (total |rate|)")
unl = np.setdiff1d(np.arange(C), learned)
OUT["mid"]["fits"] = {}
for b2, b, e in CONFIGS:
    t = tag(b2, b, e)
    y = np.abs(r[f"mid_{t}_adam"])
    p = predicted_rates(pi, BETA1, b2, e, b)
    xb0 = np.log(max(p["kink_beta2"], b * e, pi.min() * 2))
    free = fit_sbpl(unl, y, xb0=xb0)
    fix = fit_sbpl(unl, y, fixed=(1.0, 0.5), xb0=xb0)
    deepF = float(np.median(y[deep] / p["rare"][deep]))
    OUT["mid"]["fits"][t] = dict(free=dict(s1=float(free["s1"]), s2=float(free["s2"]), pi_b=free["pi_b"],
                                           rms=free["rms_log"]),
                                 fixed=dict(pi_b=fix["pi_b"], rms=fix["rms_log"]), F_rare=deepF)
    k_init = OUT["kinks"][t]["fixed"]["pi_b"]
    print(f"   {t:20s} free s1={free['s1']:.3f} s2={free['s2']:.3f} pi_b={free['pi_b']:.2e} rms={free['rms_log']:.3f} | "
          f"fixed pi_b={fix['pi_b']:.2e} (init {k_init:.2e}, ratio {fix['pi_b'] / k_init:.2f}) rms={fix['rms_log']:.3f}"
          f" | Adam/rare deepest decade {deepF:.3f}")

# --------------------------------------------------------------------------- #
# 7. psi curves
# --------------------------------------------------------------------------- #
print("\n" + "=" * 100)
print("7. psi_c at Theta_0: where does |f| = 1 - 1/C sit?")
fg = r["f_grid"]
f_own = 1 - 1 / C
OUT["psi"] = {}
for t in ["b20.99_b1_e1e-08", "b20.99_b16_e1e-08", "b20.99_b1_e0.01", "b20.99_b16_e0.01", "b20.99_b16_e0.0001",
          "b20.95_b1_e1e-08", "b20.999_b1_e1e-08"]:
    b2 = float(t.split("_")[0][3:]); b = int(t.split("_")[1][1:]); e = float(t.split("_e")[1])
    psi, dpsi = r[f"init_{t}_psi"], r[f"init_{t}_dpsi"]
    c_beta = (1 - BETA1) / np.sqrt(1 - b2)
    psat = b * c_beta * np.sqrt(2 / np.pi) / (1 - BETA1 / np.sqrt(b2))
    OUT["psi"][t] = {}
    print(f"   {t}: psi_sat = b c_beta sqrt(2/pi)/(1-beta1/sqrt(beta2)) = {psat:.3f}; 1/sqrt(eps) = {1 / np.sqrt(e):.3g}")
    print(f"     {'class':>5s} {'pi':>9s} {'n':>8s} {'psi(0)':>8s} {'pred psi(0)':>11s} {'psi(1)':>8s} {'psi(1)/psi(0)':>13s}"
          f" {'psi(1)/sat':>10s} {'elast(1)':>8s} {'f_knee':>8s}  where |f|=1 sits")
    for c in [0, 9, 49, 127]:
        d0 = dpsi[c, 0]
        p1 = np.exp(np.interp(np.log(f_own), np.log(fg[1:]), np.log(psi[c, 1:])))
        d1 = np.exp(np.interp(np.log(f_own), np.log(fg[1:]), np.log(dpsi[c, 1:])))
        Ef2 = pi[c] * (1 - 1 / C) ** 2 + (1 - pi[c]) / C ** 2
        pred0 = 1 / np.sqrt(Ef2 / b + e)
        lin = p1 / (d0 * f_own)
        el = d1 * f_own / p1
        knee = psat / d0
        if el > 0.8:
            where = "linear part"
        elif el < 0.2:
            where = "saturated part" if e < 1e-3 else "saturated"
        else:
            where = "crossover"
        n = b * pi[c] / (1 - b2)
        # does eps set the small-f slope? compare d0 with 1/sqrt(eps)
        if d0 > 0.8 / np.sqrt(e):
            where += " (slope at 0 set by eps: floor)"
        OUT["psi"][t][str(c + 1)] = dict(pi=float(pi[c]), n=float(n), dpsi0=float(d0), pred_dpsi0=float(pred0),
                                         psi1=float(p1), lin_ratio=float(lin), psi1_over_sat=float(p1 / psat),
                                         elasticity=float(el), f_knee=float(knee), psat=float(psat))
        print(f"     {c + 1:5d} {pi[c]:9.2e} {n:8.3g} {d0:8.3f} {pred0:11.3f} {p1:8.3f} {lin:13.3f} {p1 / psat:10.3f}"
              f" {el:8.3f} {knee:8.3g}  {where}")

json.dump(OUT, open(f"{OUTDIR}/results.json", "w"), indent=1, default=float)
print("\nsaved", f"{OUTDIR}/results.json")
