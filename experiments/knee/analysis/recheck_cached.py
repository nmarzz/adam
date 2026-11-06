#!/usr/bin/env python3
"""Series J, Stage 1 gate: re-verify the knee hypothesis on the cached series I ODE files.

Data: ``figs/experiments/imbalance/ode/*.npz`` (keys ``time, loss (T, C), B, O (T, C, C),
coef (T, 3, C) = diag(phi, chi, sigma)``). ``B`` and ``O`` are the logit covariance and
the class-mean logits, invariant under the ``kappa`` rescaling: the ODE integrates them in
primed units (``imbalance.solve_state``: ``r' = 5/sqrt(kappa)``, ``eta'``) and stores them
as is, starting from ``B = 0.01 kappa Id`` (``Theta_0^T Theta_0 = 0.01 Id`` in original
units). ``loss`` is ``zipf_scaling.class_metrics`` (2^14 draws).

Checks (each prints PASS / FAIL with numbers; thresholds are fixed up front):

(a) CLIFF   Adam, alpha = 1, kappa in {1, 0.1}, C in {32, 128}, all beta2.
(b) EPS     Adam, kappa = 0.1: R_c -> sqrt(eps), chi_cc/pi_c = -E[g|c]/sqrt(eps), ell pi t.
(c) TAIL    kappa = 0.1: R_A/R_S vs G sqrt(kappa)(eta_S/eta_A) sqrt(eps).
(d) K2      kappa = 1, C = 128, alpha = 1, beta2 = 0.95 (and 0.9): E4 balance ratio.
(e) SHAPE   ell_c vs L_C(m_c, s_c^2) for Adam and SGD; gamma_c.

Usage: ``real/bin/python experiments/knee/analysis/recheck_cached.py [--only a b ...]``.
Writes ``figs/experiments/knee/analysis/recheck_cached.json``.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import diagnostics as D  # noqa: E402
import lib  # noqa: E402  (experiments/imbalance/analysis/lib.py)

REPO = HERE.parents[2]
ODE = REPO / "figs" / "experiments" / "imbalance" / "ode"
OUT = REPO / "figs" / "experiments" / "knee" / "analysis"
BETA2S = (0.9, 0.95, 0.99, 0.999)
EPS = 1e-8
RESULTS = {}
VERDICTS = []


def verdict(name, ok, detail):
    VERDICTS.append((name, bool(ok), detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}", flush=True)


def load(opt, kappa, C, alpha, beta2=0.0, eta=None):
    """Longest cached base-``eta`` ODE (``lib.load_ode``), or an explicit ``eta`` file."""
    if eta is None:
        return lib.load_ode("m0", opt, kappa, C, alpha, beta2)
    opt_s = (f"b10.9_b2{beta2:g}_b1_e1e-08" if opt == "adam" else "b1")
    pat = f"m0_{opt}_k{kappa:g}_C{C}_a{alpha:g}_{opt_s}_eta{eta:g}_*"
    files = sorted(ODE.glob(pat + ".npz"))
    # prefer the rerun with the early first record (``_tm``) when there is one
    tm = [f for f in files if "_tm" in f.name]
    files = tm or files
    if not files:
        return None
    best = max(files, key=lambda f: float(f.name.split("_T")[-1].split("_s")[0]))
    out = dict(np.load(best))
    out["file"] = best.name
    return out


def ols(x, y):
    A = np.stack([x, np.ones_like(x)], 1)
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    res = y - A @ coef
    ss = np.sum((y - y.mean()) ** 2)
    return coef[0], coef[1], 1 - np.sum(res ** 2) / ss if ss > 0 else np.nan


def pct(x, q=(0, 50, 100)):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return [float(v) for v in np.percentile(x, q)] if len(x) else [np.nan] * len(q)


def fmt(v, n=3):
    return "[" + ", ".join(f"{x:.{n}g}" for x in v) + "]"


# --------------------------------------------------------------------------- #
# (a) CLIFF
# --------------------------------------------------------------------------- #
def cliff_stats(d, kappa):
    t, L, O = d["time"], d["loss"], d["O"]
    r = D.r_primed(kappa)
    m = D.margins(O)[0]
    s2 = D.diff_var(d["B"])
    gam = D.geometry(d["B"], O, r)["gamma"]
    C = L.shape[1]
    rows = []
    for c in range(C):
        ell = L[:, c]
        tc = lib.first_crossing(t, ell, 0.5 * ell[0])
        row = {"c": c, "tc": tc}
        if not np.isfinite(tc):
            rows.append(row)
            continue
        e = lib.local_exponent(t, ell)
        k = int(np.nanargmax(e))
        row["t_knee"], row["e_knee"], row["ell_knee_rel"] = t[k], e[k], ell[k] / ell[0]
        row["m_knee"] = m[k, c]
        row["knee_interior"] = k < len(t) - 3 and t[k] > tc
        after = np.flatnonzero((np.arange(len(t)) > k) & (e <= e[k] / 2))
        if len(after):
            row["t_half"], row["ell_half_rel"] = t[after[0]], ell[after[0]] / ell[0]
        # state-matched margin rates: 2 / (time from m = 1 to m = 3), 3 / (m = 3 to 6)
        tm = lambda x: lib.first_crossing(t, -m[:, c], -x)
        row["v_m13"] = 2.0 / (tm(3.0) - tm(1.0))
        row["v_m36"] = 3.0 / (tm(6.0) - tm(3.0))
        v = np.gradient(m[:, c], t)
        v0 = m[1, c] / t[1]
        row["v0"] = v0
        upto = (t > 0) & (t <= t[k])
        row["v_ratio_range_all"] = [float(np.min(v[upto] / v0)), float(np.max(v[upto] / v0))]
        win = (t >= tc) & (t <= t[k])
        row["n_win"] = int(win.sum())
        if row["knee_interior"] and win.sum() >= 3:
            row["v_ratio_range_cliff"] = [float(np.min(v[win] / v0)), float(np.max(v[win] / v0))]
            row["v_cliff"] = (np.interp(t[k], t, m[:, c]) - np.interp(tc, t, m[:, c])) / (t[k] - tc)
            row["span"] = t[k] / tc
            tt = np.linspace(tc, t[k], 41)
            y = np.interp(np.log(tt), np.log(t[1:]), np.log(ell[1:]))
            mm = np.interp(np.log(tt), np.log(t[1:]), m[1:, c])
            sl, _, row["R2_lnl_t"] = ols(tt, y)
            _, _, row["R2_lnl_logt"] = ols(np.log(tt), y)
            _, _, row["R2_m_t"] = ols(tt, mm)
            _, _, row["R2_m_logt"] = ols(np.log(tt), mm)
            ee = e[win]
            ok = ee > 0
            row["exp_growth"] = ols(np.log(t[win][ok]), np.log(ee[ok]))[0] if ok.sum() >= 3 else np.nan
            # P1: -d ln ell/dt = (1 - gamma m / r'^2) v0
            fac = np.mean((1 - gam[win, c] * m[win, c] / r ** 2))
            row["P1_ratio"] = -sl / (fac * v0)
            row["slope_lnl_t"] = -sl
        rows.append(row)
    return rows


def check_cliff():
    print("\n=== (a) CLIFF ===", flush=True)
    out = {}
    summary = []
    for C in (32, 128):
        kink = {b: lib.kink(b, C) for b in BETA2S}
        for beta2 in BETA2S:
            per_kappa = {}
            for kappa in (1.0, 0.1):
                d = load("adam", kappa, C, 1.0, beta2)
                if d is None:
                    continue
                rows = cliff_stats(d, kappa)
                per_kappa[kappa] = rows
                pri = d["priors"]
                good = [r for r in rows if r.get("R2_lnl_t") is not None and np.isfinite(r.get("R2_lnl_t", np.nan))]
                win_cls = [r for r in good if pri[r["c"]] >= kink[beta2]]
                lin_wins = np.mean([r["R2_lnl_t"] > r["R2_lnl_logt"] for r in good]) if good else np.nan
                mlin_wins = np.mean([r["R2_m_t"] > r["R2_m_logt"] for r in good]) if good else np.nan
                within = [max(abs(r["v_ratio_range_cliff"][0] - 1), abs(r["v_ratio_range_cliff"][1] - 1)) for r in good]
                within_all = [max(abs(r["v_ratio_range_all"][0] - 1), abs(r["v_ratio_range_all"][1] - 1)) for r in good]
                rec = {"n_learned": int(sum(np.isfinite(r["tc"]) for r in rows)),
                       "n_cliff": len(good), "n_window_classes": len(win_cls),
                       "frac_lnl_linear_in_t_wins": lin_wins, "frac_m_linear_in_t_wins": mlin_wins,
                       "R2_lnl_t": pct([r["R2_lnl_t"] for r in good]),
                       "R2_lnl_logt": pct([r["R2_lnl_logt"] for r in good]),
                       "exp_growth": pct([r["exp_growth"] for r in good]),
                       "span": pct([r["span"] for r in good]),
                       "v_dev_cliff": pct(within), "v_dev_all": pct(within_all),
                       "frac_v_within_25_cliff": float(np.mean(np.array(within) <= 0.25)) if good else np.nan,
                       "frac_v_within_25_all": float(np.mean(np.array(within_all) <= 0.25)) if good else np.nan,
                       "P1_ratio": pct([r["P1_ratio"] for r in good]),
                       "ell_knee_rel": pct([r["ell_knee_rel"] for r in good]),
                       "ell_half_rel": pct([r.get("ell_half_rel", np.nan) for r in good]),
                       "m_knee": pct([r["m_knee"] for r in good]),
                       "T": float(d["time"][-1])}
                out[f"C{C}_b{beta2:g}_k{kappa:g}"] = rec
                print(f"C={C} beta2={beta2:g} kappa={kappa:g}: learned {rec['n_learned']}, cliff resolved"
                      f" {rec['n_cliff']} (window-regime {rec['n_window_classes']}); ln ell linear-in-t wins"
                      f" {lin_wins:.0%}, m linear-in-t wins {mlin_wins:.0%}; R2(t) {fmt(rec['R2_lnl_t'])}"
                      f" R2(ln t) {fmt(rec['R2_lnl_logt'])}; d ln e/d ln t {fmt(rec['exp_growth'])};"
                      f" span t_knee/t_c {fmt(rec['span'])}\n    |v/v0 - 1| max over [t_c, knee]"
                      f" {fmt(rec['v_dev_cliff'])} ({rec['frac_v_within_25_cliff']:.0%} within 25%),"
                      f" over (0, knee] {fmt(rec['v_dev_all'])} ({rec['frac_v_within_25_all']:.0%});"
                      f" P1 ratio {fmt(rec['P1_ratio'])}; ell_knee/ell0 {fmt(rec['ell_knee_rel'])};"
                      f" m_knee {fmt(rec['m_knee'])}; ell/ell0 where e falls to half its max"
                      f" {fmt(rec['ell_half_rel'])}", flush=True)
                summary.append((C, beta2, kappa, rec))
            if 1.0 in per_kappa and 0.1 in per_kappa:
                pri = lib.load_ode("m0", "adam", 1.0, C, 1.0, beta2)["priors"]
                a, b = per_kappa[1.0], per_kappa[0.1]
                r0 = [b[c]["v0"] / a[c]["v0"] for c in range(C) if "v0" in a[c] and "v0" in b[c]]
                rc = [b[c]["v_cliff"] / a[c]["v_cliff"] for c in range(C)
                      if "v_cliff" in a[c] and "v_cliff" in b[c] and pri[c] >= kink[beta2]]
                rc_all = [b[c]["v_cliff"] / a[c]["v_cliff"] for c in range(C)
                          if "v_cliff" in a[c] and "v_cliff" in b[c]]
                wcls = [c for c in range(C) if pri[c] >= kink[beta2]]
                r13 = [b[c].get("v_m13", np.nan) / a[c].get("v_m13", np.nan) for c in wcls]
                r36 = [b[c].get("v_m36", np.nan) / a[c].get("v_m36", np.nan) for c in wcls]
                out[f"C{C}_b{beta2:g}_kappa_ratio"] = {"v0": pct(r0), "v_cliff_window": pct(rc),
                                                       "v_cliff_all": pct(rc_all), "n_window": len(rc),
                                                       "v_m13_window": pct(r13), "v_m36_window": pct(r36)}
                print(f"  state-matched (window classes, n={len(wcls)}): rate over m 1->3 {fmt(pct(r13))},"
                      f" m 3->6 {fmt(pct(r36))}", flush=True)
                print(f"  kappa 0.1/1 margin-rate ratio (pred sqrt(10) = 3.162): Theta0 {fmt(pct(r0), 4)};"
                      f" cliff mean, window classes (n={len(rc)}) {fmt(pct(rc))}; all resolved"
                      f" (n={len(rc_all)}) {fmt(pct(rc_all))}", flush=True)
    # SGD: m ~ log t after t_c
    for C in (32, 128):
        for kappa in (1.0, 0.1):
            d = load("sgd", kappa, C, 1.0)
            t, L, m = d["time"], d["loss"], D.margins(d["O"])[0]
            r2_log, r2_lin, slopes, late = [], [], [], []
            for c in range(C):
                tc = lib.first_crossing(t, L[:, c], 0.5 * L[0, c])
                if not np.isfinite(tc) or tc > t[-1] / 10:
                    continue
                w = t >= tc
                b, _, rl = ols(np.log(t[w]), m[w, c])
                _, _, rt = ols(t[w], m[w, c])
                r2_log.append(rl), r2_lin.append(rt), slopes.append(b)
                late.append(lib.late_slopes(t, L[:, c:c + 1], 1.0)[0])
            # Adam over the same post-t_c span, for contrast
            rec = {"n": len(r2_log), "R2_m_logt": pct(r2_log), "R2_m_t": pct(r2_lin),
                   "dm_dlnt": pct(slopes), "late_exponent": pct(late)}
            out[f"sgd_C{C}_k{kappa:g}"] = rec
            print(f"SGD C={C} kappa={kappa:g} (n={rec['n']}, t >= t_c): R2(m ~ ln t) {fmt(rec['R2_m_logt'])},"
                  f" R2(m ~ t) {fmt(rec['R2_m_t'])}, dm/dln t {fmt(rec['dm_dlnt'])},"
                  f" late ell exponent {fmt(rec['late_exponent'])}", flush=True)
    RESULTS["cliff"] = out
    # verdicts (beta2 = 0.99 required, all reported)
    for C in (32, 128):
        for kappa in (1.0, 0.1):
            rec = out.get(f"C{C}_b0.99_k{kappa:g}")
            if rec is None:
                continue
            ok = rec["frac_lnl_linear_in_t_wins"] >= 0.8
            verdict(f"(a) cliff ln ell linear in t, C={C} kappa={kappa:g} beta2=0.99", ok,
                    f"linear-in-t fit beats ln-t fit for {rec['frac_lnl_linear_in_t_wins']:.0%} of"
                    f" {rec['n_cliff']} classes (need >= 80%); median d ln e/d ln t = {rec['exp_growth'][1]:.2f}"
                    f" (1 = linear in t, 0 = power law)")
            ok = rec["frac_v_within_25_cliff"] >= 0.8
            verdict(f"(a) dm/dt constant up to knee, C={C} kappa={kappa:g} beta2=0.99", ok,
                    f"{rec['frac_v_within_25_cliff']:.0%} of classes keep v/v0 within 25% on [t_c, knee]"
                    f" (median max dev {rec['v_dev_cliff'][1]:.2f}); on (0, knee]"
                    f" {rec['frac_v_within_25_all']:.0%} (median max dev {rec['v_dev_all'][1]:.2f})")
        rr = out.get(f"C{C}_b0.99_kappa_ratio")
        if rr and rr["n_window"]:
            ok = abs(rr["v_cliff_window"][1] / np.sqrt(10) - 1) <= 0.15
            verdict(f"(a) cliff margin-rate ratio kappa 0.1/1, C={C} beta2=0.99", ok,
                    f"median {rr['v_cliff_window'][1]:.3f} (range {rr['v_cliff_window'][0]:.2f}-"
                    f"{rr['v_cliff_window'][2]:.2f}, n={rr['n_window']}) vs sqrt(10) = 3.162 (+-15%);"
                    f" state-matched m 1->3: {fmt(rr['v_m13_window'])}, m 3->6: {fmt(rr['v_m36_window'])}")


# --------------------------------------------------------------------------- #
# (b) EPS FLOOR
# --------------------------------------------------------------------------- #
def check_eps():
    print("\n=== (b) EPS FLOOR (kappa = 0.1) ===", flush=True)
    out = {}
    kappa = 0.1
    r = D.r_primed(kappa)
    eta_eff = np.sqrt(kappa) * 0.3 / np.sqrt(EPS)
    q_pred = 1 / (eta_eff * r ** 2)
    for C in (32, 128):
        for alpha in (1.0, 2.0):
            for beta2 in BETA2S:
                d = load("adam", kappa, C, alpha, beta2)
                t, pri = d["time"], d["priors"]
                idx = np.flatnonzero(t >= 8.0)
                own_bg, Rn, g1, g1se, chi_ratio, regime_eps = [], [], [], [], [], []
                for k in idx:
                    mom = D.class_moments(d["B"][k], d["O"][k])
                    nm = D.normaliser(mom, pri, beta2=beta2)
                    own_bg.append(nm["own"] + nm["bg"])
                    Rn.append(nm["R"])
                    g1.append(mom["g1"]), g1se.append(mom["g1_se"])
                    chi_ratio.append((-d["coef"][k, 1] / pri) / (mom["g1"] / np.sqrt(EPS)))
                own_bg, Rn, g1, g1se, chi_ratio = map(np.array, (own_bg, Rn, g1, g1se, chi_ratio))
                frac = own_bg / EPS                                    # (records, C)
                k8, kT = 0, len(idx) - 1
                learned = d["loss"][-1] <= 0.5 * d["loss"][0]
                sel = (frac < 0.1) & (g1se / g1 < 0.05)
                Q = d["loss"] * pri[None, :] * t[:, None]
                kq = int(np.argmin(np.abs(t - t[-1] / np.sqrt(10))))
                in_eps_T = frac[kT] < 0.1
                rec = {"T": float(t[-1]), "t_first": float(t[idx[0]]),
                       "n_eps_t8": int((frac[k8] < 1).sum()), "n_eps01_t8": int((frac[k8] < 0.1).sum()),
                       "n_eps_T": int((frac[kT] < 1).sum()), "n_eps01_T": int(in_eps_T.sum()),
                       "n_learned_T": int(learned.sum()),
                       "R_over_sqrt_eps_t8": pct(Rn[k8] / np.sqrt(EPS)),
                       "R_over_sqrt_eps_T": pct(Rn[kT] / np.sqrt(EPS)),
                       "ownbg_over_eps_T": pct(frac[kT]),
                       "chi_ratio": pct(chi_ratio[sel], (5, 50, 95)), "n_chi": int(sel.sum()),
                       "Q_T_eps_classes": pct(Q[-1, in_eps_T], (0, 50, 100)),
                       "Q_T_over_Q_Tsqrt10": pct(Q[-1, in_eps_T] / Q[kq, in_eps_T]),
                       "Q_pred_sgd_limit": q_pred}
                # t_eps: first record from which every learned class has own+bg < eps
                allin = np.all((frac < 1) | ~learned[None, :], axis=1)
                rec["t_all_learned_in_eps"] = float(t[idx][np.argmax(allin)]) if allin.any() else None
                out[f"C{C}_a{alpha:g}_b{beta2:g}"] = rec
                q = rec["Q_T_eps_classes"]
                print(f"C={C} a={alpha:g} b2={beta2:g} (T={rec['T']:.0f}): own+bg<eps for"
                      f" {rec['n_eps_t8']}/{C} classes at t={rec['t_first']:.2g}, {rec['n_eps_T']}/{C} at T"
                      f" (<0.1 eps: {rec['n_eps01_T']}; learned {rec['n_learned_T']}); R/sqrt(eps) at T"
                      f" {fmt(rec['R_over_sqrt_eps_T'], 4)}; all learned in eps from t={rec['t_all_learned_in_eps']}"
                      f"\n    chi_cc/pi_c / (-E[g|c]/sqrt eps) 5/50/95% {fmt(rec['chi_ratio'], 4)} (n={rec['n_chi']});"
                      f" ell pi t at T over eps classes min/med/max {fmt(q)} (max/min {q[2]/q[0]:.2f});"
                      f" Q(T)/Q(T/sqrt10) {fmt(rec['Q_T_over_Q_Tsqrt10'])}; SGD-limit 1/(eta_eff r'^2) = {q_pred:.3g}",
                      flush=True)
    RESULTS["eps"] = out
    for C in (32, 128):
        rec = out[f"C{C}_a1_b0.99"]
        ok = rec["n_eps_t8"] == C
        verdict(f"(b) own+bg < eps already for t >~ 8, kappa=0.1, C={C} a=1 b2=0.99", ok,
                f"{rec['n_eps_t8']}/{C} classes at t={rec['t_first']:.2g}; every class only from"
                f" t={rec['t_all_learned_in_eps']:.3g}")
        ok = rec["n_eps_T"] == C and rec["R_over_sqrt_eps_T"][2] < 1.05
        verdict(f"(b) R_c -> sqrt(eps) by T at kappa=0.1, C={C} a=1 b2=0.99", ok,
                f"{rec['n_eps_t8']}/{C} classes own+bg<eps at t={rec['t_first']:.2g}, {rec['n_eps_T']}/{C} at T;"
                f" max R/sqrt(eps) at T = {rec['R_over_sqrt_eps_T'][2]:.4f}")
        ok = abs(rec["chi_ratio"][1] - 1) < 0.05 and rec["chi_ratio"][2] - rec["chi_ratio"][0] < 0.2
        verdict(f"(b) chi_cc/pi_c = -E[g|c]/sqrt(eps), C={C} a=1 b2=0.99", ok,
                f"ratio 5/50/95% = {fmt(rec['chi_ratio'], 4)} over {rec['n_chi']} (record, class) pairs")
        q = rec["Q_T_eps_classes"]
        ok = q[2] / q[0] < 2.0
        verdict(f"(b) ell_c pi_c t const across classes, C={C} a=1 b2=0.99", ok,
                f"max/min across classes at T = {q[2]/q[0]:.2f} (need < 2); median {q[1]:.3g} vs SGD-limit {q_pred:.3g}")


# --------------------------------------------------------------------------- #
# (c) TAIL RATIO
# --------------------------------------------------------------------------- #
def risk(d):
    return d["loss"] @ d["priors"]


def interp_loglog(t, y, tq):
    return float(np.exp(np.interp(np.log(tq), np.log(t[1:]), np.log(y[1:]))))


def check_tail():
    print("\n=== (c) TAIL RATIO (kappa = 0.1) ===", flush=True)
    out = {}
    kappa = 0.1
    sets = (("base", 0.3, 3.0), ("tuned", 10.0, 300.0), ("mid", 3.0, 30.0))
    for name, eA, eS in sets:
        pred = np.sqrt(kappa) * (eS / eA) * np.sqrt(EPS)
        for C in (8, 32, 128):
            for alpha in (1.0, 2.0):
                ds = load("sgd", kappa, C, alpha, eta=eS) if name != "base" else load("sgd", kappa, C, alpha)
                if ds is None:
                    continue
                Gs = []
                for beta2 in BETA2S:
                    da = load("adam", kappa, C, alpha, beta2, eta=eA) if name != "base" else load("adam", kappa, C, alpha, beta2)
                    if da is None:
                        continue
                    tA = da["time"]
                    RA = risk(da)
                    T = tA[-1]
                    ratios = {f"{tq:.3g}": interp_loglog(tA, RA, tq) / interp_loglog(ds["time"], risk(ds), tq)
                              for tq in (T / 10, T / np.sqrt(10), T)}
                    G = ratios[f"{T:.3g}"] / pred
                    expA = lib.local_exponent(tA, RA)[-1]
                    expS = lib.local_exponent(ds["time"], risk(ds))
                    expS = float(np.interp(np.log(T), np.log(ds["time"][1:]), expS[1:]))
                    rec = {"T": float(T), "ratio": ratios, "pred": pred, "G": G,
                           "adam_risk_exponent_T": float(expA), "sgd_risk_exponent_T": expS}
                    out[f"{name}_C{C}_a{alpha:g}_b{beta2:g}"] = rec
                    Gs.append(G)
                    print(f"{name} (eta_A={eA:g}, eta_S={eS:g}) C={C} a={alpha:g} b2={beta2:g}: R_A/R_S at"
                          f" T/10, T/sqrt10, T={T:.0f}: " + ", ".join(f"{v:.3g}" for v in ratios.values())
                          + f"; pred/G={pred:.3g}; G={G:.2f}; risk exponents at T: Adam {expA:.2f}, SGD {expS:.2f}",
                          flush=True)
                if Gs:
                    out[f"{name}_C{C}_a{alpha:g}_Gspread"] = [min(Gs), max(Gs)]
    RESULTS["tail"] = out
    for name in ("base", "tuned"):
        for C in (32, 128):
            for alpha in (1.0, 2.0):
                Gs = [out[k]["G"] for k in out if k.startswith(f"{name}_C{C}_a{alpha:g}_b")]
                if not Gs:
                    continue
                ok = min(Gs) >= 1.0 and max(Gs) <= 1.4
                tail = [k for k in out if k.startswith(f"{name}_C{C}_a{alpha:g}_b")
                        and 0.9 <= out[k]["adam_risk_exponent_T"] <= 1.1]
                Gt = [out[k]["G"] for k in tail]
                verdict(f"(c) tail ratio G in [1, 1.4], {name} C={C} a={alpha:g}", ok,
                        f"G over beta2 0.9/0.95/0.99/0.999 = {fmt(Gs)} (spread max/min {max(Gs)/min(Gs):.2f});"
                        f" restricted to runs whose Adam risk exponent at T is in [0.9, 1.1] (in a 1/t tail):"
                        f" {len(Gt)}/{len(Gs)} runs, G = {fmt(Gt) if Gt else '-'}")


# --------------------------------------------------------------------------- #
# (d) K2 BALANCE
# --------------------------------------------------------------------------- #
def gap_stationarity_per_class(t, O, frac=1 / 3):
    k0 = int(np.argmin(np.abs(t - t[-1] * frac)))
    G = lambda o: np.diagonal(o)[None, :] - o
    a, b = G(O[k0]), G(O[-1])
    return np.abs(b - a).max(0) / np.abs(b).max(0), float(t[k0])


def check_k2(beta2s=(0.95, 0.9)):
    print("\n=== (d) K2 BALANCE (kappa = 1, C = 128, alpha = 1) ===", flush=True)
    import frozen
    out = {}
    kappa, C, alpha = 1.0, 128, 1.0
    r2 = D.r_primed(kappa) ** 2
    eta = 0.3 * np.sqrt(kappa)
    for beta2 in beta2s:
        d = load("adam", kappa, C, alpha, beta2)
        t, pri = d["time"], d["priors"]
        B, O = d["B"][-1], d["O"][-1]
        t0 = time.time()
        co = frozen.coefficients(B, O, pri, beta2)
        compile_s = time.time() - t0
        diag_err = np.max(np.abs(np.stack([np.diag(co["phi"]), np.diag(co["chi"]),
                                            np.diag(co["sigma"])]) - d["coef"][-1])
                          / np.abs(d["coef"][-1]))
        bal = D.balance(O, co["phi"], co["chi"], r2)
        stat, t13 = gap_stationarity_per_class(t, d["O"])
        learned = d["loss"][-1] <= 0.5 * d["loss"][0]
        m = D.margins(d["O"])[0]
        k13 = int(np.argmin(np.abs(t - t[-1] / 3)))
        dm_dt = (m[-1] - m[k13]) / (t[-1] - t[k13])
        rel_drift = np.abs(dm_dt) / (eta * bal["forcing"])
        ito = D.ito_split(B, O, co["phi"], co["chi"], co["sigma"], r2=r2, eta=eta)
        dev = np.abs(bal["ratio"] - 1)
        rec = {"T": float(t[-1]), "t_third": t13, "coef_diag_max_rel_err_vs_stored": float(diag_err),
               "seconds_first_call": compile_s, "n_learned": int(learned.sum()),
               "ratio": pct(bal["ratio"]), "ratio_plain": pct(bal["ratio_plain"]),
               "ratio_learned": pct(bal["ratio"][learned]), "ratio_unlearned": pct(bal["ratio"][~learned]),
               "frac_within_20": float(np.mean(dev <= 0.2)), "max_abs_ratio_minus_1": float(dev.max()),
               "plain_max_abs_ratio_minus_1": float(np.abs(bal["ratio_plain"] - 1).max()),
               "worst_classes": [int(c) for c in np.argsort(-dev)[:5]],
               "worst_ratios": bal["ratio"][np.argsort(-dev)[:5]].tolist(),
               "mbar_over_m": pct(bal["mbar"] / m[-1]),
               "lambda_over_forcing": pct(bal["lambda"] / bal["forcing"]),
               "stationarity_T3": pct(stat), "stationarity_global_T3": float(stat.max()),
               "gap_stationarity_lib_Tsqrt10": lib.gap_stationarity(t, d["O"]),
               "dm_dt_over_eta_forcing": pct(rel_drift),
               "margin_rate_over_forcing_frozen": pct(np.abs(bal["margin_rate"]) / bal["forcing"]),
               "ito_share_of_dl_dt": pct(np.abs(ito["ito"]) / (np.abs(ito["O"]) + np.abs(ito["B_drift"]) + np.abs(ito["ito"]))),
               "dl_dt_parts_median": {k: float(np.median(v)) for k, v in ito.items()},
               "per_class": {"ratio": bal["ratio"].tolist(), "mbar": bal["mbar"].tolist(),
                             "mbar_pred": bal["mbar_pred"].tolist(), "m": m[-1].tolist(),
                             "lambda": bal["lambda"].tolist(), "forcing": bal["forcing"].tolist(),
                             "phi_cc": bal["phi_cc"].tolist(), "stationarity": stat.tolist(),
                             "learned": learned.tolist()}}
        out[f"b{beta2:g}"] = rec
        print(f"beta2={beta2:g} T={rec['T']:.0f}: coefficient diag vs stored max rel err {diag_err:.2e};"
              f" learned {rec['n_learned']}/{C}\n    balance ratio phi mbar/(r'^2|chi|+lambda) min/med/max"
              f" {fmt(rec['ratio'], 4)} (learned {fmt(rec['ratio_learned'], 4)}, unlearned"
              f" {fmt(rec['ratio_unlearned'], 4)}); max |ratio - 1| = {rec['max_abs_ratio_minus_1']:.2e};"
              f" within 20%: {rec['frac_within_20']:.0%}; worst classes"
              f" {rec['worst_classes']} -> {fmt(rec['worst_ratios'], 4)}\n    plain-margin ratio"
              f" {fmt(rec['ratio_plain'], 4)}; mbar/m {fmt(rec['mbar_over_m'])}; lambda/forcing"
              f" {fmt(rec['lambda_over_forcing'])}\n    gap stationarity per class [T/3, T]"
              f" {fmt(rec['stationarity_T3'])} (global {rec['stationarity_global_T3']:.2e}; lib T/sqrt10:"
              f" {rec['gap_stationarity_lib_Tsqrt10']:.2e}); |dm/dt|/(eta r'^2|chi|) {fmt(rec['dm_dt_over_eta_forcing'])};"
              f" frozen |margin rate|/forcing {fmt(rec['margin_rate_over_forcing_frozen'])}\n    Ito share of"
              f" |d ell/dt| {fmt(rec['ito_share_of_dl_dt'])}; median parts {rec['dl_dt_parts_median']}",
              flush=True)
    RESULTS["k2"] = out
    rec = out["b0.95"]
    verdict("(d) K2 balance ratio within 20% for every class (kappa=1 C=128 a=1 b2=0.95)",
            rec["frac_within_20"] == 1.0,
            f"{rec['frac_within_20']:.0%} of classes within 20%; max |ratio - 1| = {rec['max_abs_ratio_minus_1']:.1e}"
            f" (weighted margin mbar), {rec['plain_max_abs_ratio_minus_1']:.3f} with the plain margin m")
    verdict("(d) gaps stationary on [T/3, T] (kappa=1 C=128 a=1 b2=0.95)",
            rec["stationarity_global_T3"] < 0.01,
            f"max over classes of max_k |dG_kc|/max_k |G_kc| = {rec['stationarity_global_T3']:.2e}"
            f" (need < 1e-2); per class min/med/max {fmt(rec['stationarity_T3'])}")


# --------------------------------------------------------------------------- #
# (e) SHAPE
# --------------------------------------------------------------------------- #
def check_shape():
    print("\n=== (e) SHAPE: ell_c vs L_C(m_c, s_c^2) ===", flush=True)
    out = {}
    rng = np.random.default_rng(0)
    for C in (32, 128):
        runs = {}
        for kappa in (1.0, 0.1):
            for beta2 in (0.9, 0.99, 0.999):
                runs[("adam", kappa, beta2)] = load("adam", kappa, C, 1.0, beta2)
            runs[("sgd", kappa, 0.0)] = load("sgd", kappa, C, 1.0)
        geo = {k: (D.margins(d["O"])[0], D.diff_var(d["B"])) for k, d in runs.items()}
        allm = np.concatenate([g[0].ravel() for g in geo.values()])
        alls = np.concatenate([g[1].ravel() for g in geo.values()])
        t0 = time.time()
        table = D.ShapeTable(C, float(np.sqrt(alls.max())) * 1.02, float(allm.min()) - 1,
                             float(allm.max()) + 1)
        # interpolation check against direct evaluations at random data points
        pick = rng.choice(len(allm), 40, replace=False)
        direct = np.array([D.shape_loss(C, allm[i], alls[i])[()] for i in pick])
        interp = table(allm[pick], alls[pick])
        terr = float(np.max(np.abs(np.log(interp / direct))))
        print(f"C={C}: shape table {time.time() - t0:.0f}s, max |log interp/direct| = {terr:.1e}", flush=True)
        out[f"C{C}_table_err"] = terr
        bins = ((0.5, 1.01), (1e-2, 0.5), (1e-4, 1e-2), (0, 1e-4))
        for key, d in runs.items():
            opt, kappa, beta2 = key
            m, s2 = geo[key]
            ell = d["loss"][1:]
            Lc = table(m[1:], s2[1:])
            res = np.log(ell / Lc)
            rel = ell / d["loss"][0][None, :]
            g = D.geometry(d["B"], d["O"], D.r_primed(kappa))
            rec = {"median_res": float(np.median(res)), "median_abs": float(np.median(np.abs(res))),
                   "p90_abs": float(np.percentile(np.abs(res), 90))}
            for lo, hi in bins:
                sel = (rel > lo) & (rel <= hi)
                rec[f"bin_{lo:g}_{hi:g}"] = (pct(res[sel], (10, 50, 90)) + [int(sel.sum())]) if sel.any() else None
            # gamma at t_c and at T (learned classes); asymptotic form ln L = ln(C-1) - m + s^2/2
            tcs = lib.level_times(d["time"], d["loss"])
            kc = [int(np.argmin(np.abs(d["time"] - tc))) for tc in tcs if np.isfinite(tc)]
            cl = [c for c, tc in enumerate(tcs) if np.isfinite(tc)]
            rec["gamma_tc"] = pct([g["gamma"][k, c] for k, c in zip(kc, cl)])
            rec["gamma_T"] = pct(g["gamma"][-1, cl])
            rec["noise_share_T"] = pct((g["s2_noise"][-1] / g["s2"][-1])[cl])
            small = rel < 1e-2
            asym = np.log(C - 1) - m[1:] + s2[1:] / 2
            rec["asym_minus_exact_small"] = pct((asym - np.log(Lc))[small], (10, 50, 90)) if small.any() else None
            out[f"C{C}_{opt}_k{kappa:g}_b{beta2:g}"] = rec
            print(f"  {opt:4s} kappa={kappa:g} b2={beta2:g}: ln(ell/L_C) median {rec['median_res']:+.3f},"
                  f" median|.| {rec['median_abs']:.3f}, p90|.| {rec['p90_abs']:.3f}; by ell/ell0 bin (10/50/90%, n): "
                  + "; ".join(f"{lo:g}-{hi:g}: {fmt(rec[f'bin_{lo:g}_{hi:g}'][:3], 2)}" for lo, hi in bins
                              if rec[f"bin_{lo:g}_{hi:g}"])
                  + f"\n      gamma at t_c {fmt(rec['gamma_tc'])}, at T {fmt(rec['gamma_T'])}; noise share of s^2 at T"
                  f" {fmt(rec['noise_share_T'])}; (ln(C-1)-m+s^2/2) - ln L_C for ell<1e-2: "
                  f"{fmt(rec['asym_minus_exact_small'] or [np.nan])}", flush=True)
        # MC noise of the stored losses: recompute at the last state of each run
        noise = []
        for key, d in runs.items():
            mom = D.class_moments(d["B"][-1], d["O"][-1], n=2 ** 16)
            noise.append(np.abs(np.log(d["loss"][-1] / mom["loss"])))
        out[f"C{C}_stored_loss_mc_noise"] = pct(np.concatenate(noise), (50, 90, 100))
        print(f"  stored-loss MC noise at T (|ln stored/recomputed 2^16|) 50/90/100%:"
              f" {fmt(out[f'C{C}_stored_loss_mc_noise'])}", flush=True)
    RESULTS["shape"] = out
    for C in (32, 128):
        a = [out[k]["median_abs"] for k in out if k.startswith(f"C{C}_adam")]
        s = [out[k]["median_abs"] for k in out if k.startswith(f"C{C}_sgd")]
        ok = max(a + s) < 0.1
        verdict(f"(e) shape collapse ell ~ L_C(m, s^2), C={C}", ok,
                f"median |ln(ell/L_C)| Adam {fmt(a, 2)}, SGD {fmt(s, 2)} (need < 0.1 everywhere)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="+", default=["a", "b", "c", "d", "e"])
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "recheck_cached.json"
    prev = json.loads(path.read_text()) if path.exists() else {}
    t0 = time.time()
    fns = {"a": check_cliff, "b": check_eps, "c": check_tail, "d": check_k2, "e": check_shape}
    for k in args.only:
        fns[k]()
    prev.update(lib.jsonable(RESULTS))
    prev.setdefault("verdicts", {})
    for name, ok, detail in VERDICTS:
        prev["verdicts"][name] = {"pass": ok, "detail": detail}
    path.write_text(json.dumps(prev, indent=1))
    print(f"\n{sum(v[1] for v in VERDICTS)}/{len(VERDICTS)} checks pass ({time.time() - t0:.0f}s);"
          f" wrote {path}")
    for name, ok, detail in VERDICTS:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")


if __name__ == "__main__":
    main()
