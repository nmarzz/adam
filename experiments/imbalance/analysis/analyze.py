"""Series-I analysis: tables (stdout, markdown) and ``numbers.json``.

Run: ``real/bin/python experiments/imbalance.py analyze`` (or from this directory).
Every block skips configurations whose data do not exist yet, so it can run after
each stage.
"""
from __future__ import annotations

import json

import numpy as np

import lib
from lib import I

ADAM_B2 = I.BETA2S
NUM = {"runs": {}, "kink": {}, "pairs": {}, "frozen": {}, "frozen_bias": {},
       "validation": {}, "m1_vs_m0": {}, "bias": {}}


def key(*parts):
    return "_".join(str(p) for p in parts)


def opt_list():
    return [("adam", b2) for b2 in ADAM_B2] + [("sgd", 0.0)]


def label(opt, b2):
    return f"adam{b2:g}" if opt == "adam" else "sgd"


# --------------------------------------------------------------------------- #
# 1. Per-run summaries
# --------------------------------------------------------------------------- #
def per_run():
    print("\n## Runs: risk, stall, t_c\n")
    print("| model | kappa | C | alpha | opt | src | T | R(0) | R(T) | t(R/2) | t(R/10) |"
          " -dlogR/dlogt at 1/10/100/T | learned | above init | stat |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for model in I.MODELS:
        for kappa in I.KAPPAS:
            for C in I.CLASSES:
                for alpha in I.ALPHAS:
                    for opt, b2 in opt_list():
                        cur = lib.curves(model, opt, kappa, C, alpha, b2)
                        if cur is None:
                            continue
                        t, L, pi = cur["time"], cur["loss"], cur["priors"]
                        Rk = L @ pi
                        tcs = {q: lib.level_times(t, L, q) for q in lib.LEVELS}
                        ex = lib.local_exponent(t, Rk)
                        at = lambda s: float(np.interp(np.log(s), np.log(t[1:]), ex[1:]))
                        diag = lib.stall_diagnostics(t, L)
                        stat = lib.gap_stationarity(t, cur["O"]) if "O" in cur else np.nan
                        rec = {
                            "source": cur["source"], "T": float(t[-1]), "pi": pi,
                            "R0": float(Rk[0]), "RT": float(Rk[-1]),
                            "t_half": lib.first_crossing(t, Rk, Rk[0] / 2),
                            "t_tenth": lib.first_crossing(t, Rk, Rk[0] / 10),
                            "exp_at": {s: at(s) for s in (1.0, 10.0, 100.0, float(t[-1]))
                                       if s <= t[-1]},
                            "tc": {str(q): tcs[q] for q in lib.LEVELS},
                            "slope_at_tc": lib.slope_at(t, L, tcs[0.5]),
                            "late_slope": lib.late_slopes(t, L),
                            "stationarity": stat, **diag,
                        }
                        NUM["runs"][key(model, kappa, C, alpha, label(opt, b2))] = rec
                        ex_s = "/".join(f"{rec['exp_at'].get(s, np.nan):.2f}"
                                        for s in (1.0, 10.0, 100.0)) + f"/{ex[-1]:.2f}"
                        print(f"| {model} | {kappa:g} | {C} | {alpha:g} | {label(opt, b2)} |"
                              f" {cur['source']} | {t[-1]:.0f} | {Rk[0]:.3f} | {Rk[-1]:.4f} |"
                              f" {rec['t_half']:.3g} | {rec['t_tenth']:.3g} | {ex_s} |"
                              f" {diag['learned']}/{C} | {diag['above_initial']} |"
                              f" {stat:.1e} |")


# --------------------------------------------------------------------------- #
# 2. Kink: Adam/SGD t_c ratio, t_c breaks, beta2 pairs
# --------------------------------------------------------------------------- #
def kink_fits():
    print("\n## Kink from trajectories (t_c at 1/2): Adam/SGD ratio and t_c breaks\n")
    print("| model | kappa | C | alpha | beta2 | n fit (above/below) | ratio pi_b | /kink | /kink_F"
          " | level / pred rare | free t_c break (slopes) | smooth p1, p2 / kink_F |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for model in I.MODELS:
        for kappa in I.KAPPAS:
            for C in I.CLASSES:
                for alpha in I.ALPHAS:
                    sgd = NUM["runs"].get(key(model, kappa, C, alpha, "sgd"))
                    if sgd is None:
                        continue
                    pi = np.asarray(sgd["pi"])
                    ts = np.asarray(sgd["tc"]["0.5"])
                    for b2 in ADAM_B2:
                        a = NUM["runs"].get(key(model, kappa, C, alpha, f"adam{b2:g}"))
                        if a is None:
                            continue
                        ta = np.asarray(a["tc"]["0.5"])
                        ok = (lib.fit_set(pi, ta, a["T"]) & lib.fit_set(pi, ts, sgd["T"]))
                        k0, kF = lib.kink(b2), lib.kink(b2, C)
                        _, rare_pred = lib.ratio_asymptotes(b2, C, kappa)
                        rec = {"n": int(ok.sum()), "kink": k0, "kink_F": kF,
                               "rare_pred": rare_pred, "classes": np.flatnonzero(ok) + 1}
                        line = (f"| {model} | {kappa:g} | {C} | {alpha:g} | {b2:g} |")
                        if ok.sum() >= 3:
                            rb = lib.ratio_break(pi[ok], ta[ok] / ts[ok])
                            rec["ratio"] = rb
                            line += (f" {rb['n']} ({rb['n_above']}/{rb['n_below']}) |"
                                     f" {rb['pi_b']:.3g}{'' if rb['interior'] else '*'}"
                                     f"{'' if min(rb['n_above'], rb['n_below']) >= 2 else '†'} |"
                                     f" {rb['pi_b'] / k0:.2f} | {rb['pi_b'] / kF:.2f} |"
                                     f" {rb['level'] / rare_pred:.2f} |")
                            br = lib.tc_breaks(pi[ok], ta[ok])
                            rec["tc_breaks"] = br
                            fr = br.get("free")
                            line += (f" {fr['pi_b'] / kF:.2f} ({fr['a_above']:.2f}/"
                                     f"{fr['a_below']:.2f}) |" if fr else " – |")
                            line += (f" {br['smooth_p1']['pi_b'] / kF:.2f},"
                                     f" {br['smooth_p2']['pi_b'] / kF:.2f} |")
                        else:
                            line += f" {int(ok.sum())} | unresolved | | | | | |"
                        NUM["kink"][key(model, kappa, C, alpha, b2)] = rec
                        print(line)
    print("\n(* = break outside the fitted pi range; † = fewer than 2 classes on one side;"
          " kink_F = kappa_beta(1-beta2)/(b F^2))")


def beta2_pairs():
    print("\n## beta2-pair t_c ratios (median over classes; rare side pi < kink_F/3,"
          " window pi > 3 kink_F)\n")
    print("| model | kappa | C | alpha | pair | pred rare | rare: median (n) | window: median (n) |")
    print("|---|---|---|---|---|---|---|---|")
    pred = {(0.9, 0.95): lib.coef_beta(0.95) / lib.coef_beta(0.9),
            (0.95, 0.99): lib.coef_beta(0.99) / lib.coef_beta(0.95)}
    for model in I.MODELS:
        for kappa in I.KAPPAS:
            for C in I.CLASSES:
                for alpha in I.ALPHAS:
                    for (b_a, b_b), p in pred.items():
                        ra = NUM["runs"].get(key(model, kappa, C, alpha, f"adam{b_a:g}"))
                        rb = NUM["runs"].get(key(model, kappa, C, alpha, f"adam{b_b:g}"))
                        if ra is None or rb is None:
                            continue
                        pi = np.asarray(ra["pi"])
                        r = np.asarray(ra["tc"]["0.5"]) / np.asarray(rb["tc"]["0.5"])
                        ok = (lib.fit_set(pi, np.asarray(ra["tc"]["0.5"]), ra["T"])
                              & lib.fit_set(pi, np.asarray(rb["tc"]["0.5"]), rb["T"]))
                        kF = max(lib.kink(b_a, C), lib.kink(b_b, C))
                        kf = min(lib.kink(b_a, C), lib.kink(b_b, C))
                        rare = ok & (pi < kf / 3)
                        win = ok & (pi > 3 * kF)
                        med = lambda m: float(np.median(r[m])) if m.any() else np.nan
                        NUM["pairs"][key(model, kappa, C, alpha, b_a, b_b)] = {
                            "pred_rare": p, "ratio": r, "rare": med(rare), "n_rare": int(rare.sum()),
                            "window": med(win), "n_window": int(win.sum())}
                        print(f"| {model} | {kappa:g} | {C} | {alpha:g} | {b_a:g}/{b_b:g} |"
                              f" {p:.3f} | {med(rare):.3f} ({rare.sum()}) |"
                              f" {med(win):.3f} ({win.sum()}) |")


# --------------------------------------------------------------------------- #
# 3. Frozen initial drift (m0): alpha-pooled SBPL fits
# --------------------------------------------------------------------------- #
def frozen():
    print("\n## Frozen initial drift (m0, Theta_0): alpha-pooled SBPL fits\n")
    print("| kappa | C | beta2 | n | free: s_rare/s_freq | free pi_b / kink_F | fixed (1,1/2) pi_b"
          " / kink_F | alpha overlap |rel diff| | SGD / r^2 pi(1-1/C) |")
    print("|---|---|---|---|---|---|---|---|---|")
    for kappa in I.KAPPAS:
        for C in I.CLASSES:
            data = {a: lib.load_frozen(kappa, C, a) for a in I.ALPHAS}
            if any(v is None for v in data.values()):
                continue
            for b2 in ADAM_B2:
                pis = np.concatenate([data[a]["priors"] for a in I.ALPHAS])
                rate = np.abs(np.concatenate([data[a][f"init_b2{b2:g}_adam"] for a in I.ALPHAS]))
                sgd = np.abs(np.concatenate([data[a][f"init_b2{b2:g}_sgd"] for a in I.ALPHAS]))
                order = np.argsort(pis)
                pis, rate, sgd = pis[order], rate[order], sgd[order]
                kF = lib.kink(b2, C)
                n_lo, n_hi = int((pis < kF).sum()), int((pis >= kF).sum())
                free = lib.fit_sbpl(pis, rate) if min(n_lo, n_hi) >= 5 else None
                fixed = lib.fit_sbpl(pis, rate, fixed=(1.0, 0.5))
                # alpha-independence: interpolate alpha=1 onto alpha=2 classes in the overlap
                p1, r1 = data[1.0]["priors"], np.abs(data[1.0][f"init_b2{b2:g}_adam"])
                p2, r2 = data[2.0]["priors"], np.abs(data[2.0][f"init_b2{b2:g}_adam"])
                m = (p2 >= p1.min()) & (p2 <= p1.max())
                i1 = np.argsort(p1)
                interp = np.exp(np.interp(np.log(p2[m]), np.log(p1[i1]), np.log(r1[i1])))
                overlap = float(np.max(np.abs(interp / r2[m] - 1))) if m.any() else np.nan
                sgd_ratio = sgd / ((lib.R ** 2) * pis * (1 - 1 / C))
                rec = {"pi": pis, "rate": rate, "sgd": sgd, "kink_F": kF, "fixed": fixed,
                       "free": free, "alpha_overlap": overlap,
                       "sgd_ratio": [float(sgd_ratio.min()), float(sgd_ratio.max())]}
                NUM["frozen"][key(kappa, C, b2)] = rec
                fs = (f"{free['s1']:.2f}/{free['s2']:.2f} | {free['pi_b'] / kF:.2f}"
                      if free else f"(sides {n_lo}/{n_hi}) | –")
                print(f"| {kappa:g} | {C} | {b2:g} | {len(pis)} | {fs} |"
                      f" {fixed['pi_b'] / kF:.2f} (rms {fixed['rms_log']:.3f}) |"
                      f" {overlap:.3f} | {sgd_ratio.min():.3f}-{sgd_ratio.max():.3f} |")


def frozen_bias():
    print("\n## Frozen bias drift (m1, chains): E[m/sqrt(v+eps)] (Adam) and E f (SGD)\n")
    print("| kappa | C | alpha | state | beta2 | Adam drift: class 1 / median / last |"
          " SGD drift: class 1 / last | min own events | Adam sign changes at pi |")
    print("|---|---|---|---|---|---|---|---|---|")
    for kappa in I.KAPPAS:
        for C in I.CLASSES:
            for alpha in I.ALPHAS:
                d = lib.load_frozen_bias(kappa, C, alpha)
                if d is None:
                    continue
                pi = d["priors"]
                for state in ("init", "prior"):
                    for b2 in ADAM_B2:
                        u = d[f"{state}_b2{b2:g}_adam"]
                        f = d[f"{state}_b2{b2:g}_sgd"]
                        ev = d[f"{state}_b2{b2:g}_events"]
                        s = np.flatnonzero(np.diff(np.sign(u)) != 0)
                        cross = (float(np.sqrt(pi[s[0]] * pi[s[0] + 1])) if len(s) else np.nan)
                        NUM["frozen_bias"][key(kappa, C, alpha, state, b2)] = {
                            "pi": pi, "adam": u, "adam_se": d[f"{state}_b2{b2:g}_adam_se"],
                            "sgd": f, "events": ev, "cross": cross}
                        print(f"| {kappa:g} | {C} | {alpha:g} | {state} | {b2:g} |"
                              f" {u[0]:+.3f} / {np.median(u):+.3f} / {u[-1]:+.3f} |"
                              f" {f[0]:+.3f} / {f[-1]:+.4f} | {ev.min():.0f} | {cross:.3g} |")


# --------------------------------------------------------------------------- #
# 4. Validation (m0 ODE vs sims), m1 vs m0, learned bias
# --------------------------------------------------------------------------- #
def validation():
    print("\n## m0: ODE vs simulation\n")
    print("| kappa | C | alpha | opt | D | T_sim | time lag t_sim/t_ode at fixed R (t>=1): median [min, max]"
          " | sup_t>=1 |R_sim-R_ode|/R_ode (z) | mean log(t_c sim/ode) +- se (n) | learned sim/ode |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for kappa in I.KAPPAS:
        for C in I.CLASSES:
            for alpha in I.ALPHAS:
                for opt, b2 in opt_list():
                    ode = lib.load_ode("m0", opt, kappa, C, alpha, b2)
                    if ode is None:
                        continue
                    for D, T in ((1000, None), (2000, 100.0), (4000, 100.0), (4000, 300.0)):
                        sim = lib.load_sim("m0", opt, kappa, C, alpha, b2, D=D, T=T)
                        if sim is None:
                            continue
                        pi = sim["priors"]
                        t = sim["time"]
                        m = t >= 1.0
                        Rs_runs = sim["loss"] @ pi                    # (runs, T)
                        Rs = Rs_runs.mean(0)
                        se = Rs_runs.std(0, ddof=1) / np.sqrt(len(Rs_runs))
                        Ro = lib.interp_log(t, ode["time"], ode["loss"] @ pi)
                        rel = np.abs(Rs - Ro) / Ro
                        k = int(np.argmax(np.where(m, rel, -1)))
                        z = (Rs[k] - Ro[k]) / max(se[k], 1e-300)
                        tc_s = lib.level_times(t, sim["loss"].mean(0), 0.5)
                        tc_o = lib.level_times(ode["time"], ode["loss"], 0.5)
                        ok = np.isfinite(tc_s) & np.isfinite(tc_o) & (tc_o >= 1) & (tc_o <= t[-1] / 2)
                        lr = np.log(tc_s[ok] / tc_o[ok])
                        mean_lr = float(lr.mean()) if ok.any() else np.nan
                        se_lr = float(lr.std(ddof=1) / np.sqrt(ok.sum())) if ok.sum() > 1 else np.nan
                        learned_s = int(np.isfinite(tc_s).sum())
                        k_end = int(np.argmin(np.abs(ode["time"] - t[-1])))
                        learned_o = int(np.isfinite(lib.level_times(ode["time"][:k_end + 1],
                                                                    ode["loss"][:k_end + 1])).sum())
                        _, lag = lib.time_lag(t, Rs, ode["time"], ode["loss"] @ pi)
                        lag_s = (f"{np.median(lag):.3f} [{lag.min():.3f}, {lag.max():.3f}]"
                                 if len(lag) else "–")
                        NUM["validation"][key(kappa, C, alpha, label(opt, b2), D, int(t[-1]))] = {
                            "lag_median": float(np.median(lag)) if len(lag) else np.nan,
                            "lag_min": float(lag.min()) if len(lag) else np.nan,
                            "lag_max": float(lag.max()) if len(lag) else np.nan,
                            "sup_rel": float(rel[k]), "z": float(z), "t_sup": float(t[k]),
                            "log_tc": mean_lr, "log_tc_se": se_lr, "n": int(ok.sum()),
                            "learned_sim": learned_s, "learned_ode": learned_o}
                        print(f"| {kappa:g} | {C} | {alpha:g} | {label(opt, b2)} | {D} |"
                              f" {t[-1]:.0f} | {lag_s} | {rel[k]:.3f} ({z:+.1f}) at t={t[k]:.3g} |"
                              f" {mean_lr:+.3f} +- {se_lr:.3f} ({ok.sum()}) |"
                              f" {learned_s}/{learned_o} |")


def m1_vs_m0():
    print("\n## m1 (bias) vs m0, both from simulation (D = 1000)\n")
    print("| kappa | C | alpha | opt | R(T) m0 / m1 | t(R/2) m0 / m1 | learned m0 / m1 |"
          " median t_c(m1)/t_c(m0) | bias: corr(beta-mean, log pi), slope |")
    print("|---|---|---|---|---|---|---|---|---|")
    for kappa in I.KAPPAS:
        for C in I.CLASSES:
            for alpha in I.ALPHAS:
                for opt, b2 in opt_list():
                    s0 = lib.load_sim("m0", opt, kappa, C, alpha, b2)
                    s1 = lib.load_sim("m1", opt, kappa, C, alpha, b2)
                    if s1 is None:
                        continue
                    pi, t = s1["priors"], s1["time"]
                    L1 = s1["loss"].mean(0)
                    R1 = L1 @ pi
                    beta = s1["beta"][:, -1].mean(0)
                    bc = beta - beta.mean()
                    lp = np.log(pi) - np.log(pi).mean()
                    corr = float(np.corrcoef(bc, lp)[0, 1])
                    slope = float(np.polyfit(lp, bc, 1)[0])
                    tc1 = lib.level_times(t, L1)
                    rec = {"RT_m1": float(R1[-1]), "t_half_m1": lib.first_crossing(t, R1, R1[0] / 2),
                           "learned_m1": int(np.isfinite(tc1).sum()), "beta_T": beta,
                           "beta_corr": corr, "beta_slope": slope, "pi": pi}
                    line = f"| {kappa:g} | {C} | {alpha:g} | {label(opt, b2)} |"
                    if s0 is not None and len(s0["time"]) == len(t):
                        L0 = s0["loss"].mean(0)
                        R0 = L0 @ pi
                        tc0 = lib.level_times(t, L0)
                        ok = np.isfinite(tc0) & np.isfinite(tc1)
                        med = float(np.median(tc1[ok] / tc0[ok])) if ok.any() else np.nan
                        rec.update({"RT_m0": float(R0[-1]),
                                    "t_half_m0": lib.first_crossing(t, R0, R0[0] / 2),
                                    "learned_m0": int(np.isfinite(tc0).sum()), "tc_ratio": tc1 / tc0,
                                    "median_tc_ratio": med})
                        line += (f" {R0[-1]:.4f} / {R1[-1]:.4f} | {rec['t_half_m0']:.3g} /"
                                 f" {rec['t_half_m1']:.3g} | {rec['learned_m0']} /"
                                 f" {rec['learned_m1']} | {med:.2f} |")
                    else:
                        line += f" – / {R1[-1]:.4f} | – / {rec['t_half_m1']:.3g} | – /" \
                                f" {rec['learned_m1']} | – |"
                    line += f" {corr:+.2f}, {slope:.2f} |"
                    NUM["m1_vs_m0"][key(kappa, C, alpha, label(opt, b2))] = rec
                    print(line)


def main():
    per_run()
    kink_fits()
    beta2_pairs()
    frozen()
    frozen_bias()
    validation()
    m1_vs_m0()
    (lib.OUTDIR / "numbers.json").write_text(json.dumps(lib.jsonable(NUM), indent=1))
    print(f"\nwrote {lib.OUTDIR / 'numbers.json'}")


if __name__ == "__main__":
    main()
