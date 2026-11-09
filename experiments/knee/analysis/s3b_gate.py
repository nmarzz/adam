"""S3b gate (plan, Stage 3): the series J simulation at C = 128, alpha = 1, kappa = 1, eps = 1e-8
must reproduce series I's ODE there: the beta2 <= 0.95 fixed point with ~27-31 classes learned,
and per-class half-loss times within ~3%.

Compares per-class ``t_c`` (seed-mean class loss vs the cached ODE's ``loss``) on the classes
learned in both, learned counts at ``t = 316`` and at the simulation's ``T``, and the final
loss of the stalled classes. Run: ``real/bin/python experiments/knee/analysis/s3b_gate.py``.
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np

import lib_knee as K

ODE = K.REPO / "figs" / "experiments" / "imbalance" / "ode"


def ode_file(opt, beta2):
    if opt == "sgd":
        pat = "m0_sgd_k1_C128_a1_b1_eta3_T*_s1.npz"
    else:
        pat = f"m0_adam_k1_C128_a1_b10.9_b2{beta2:g}_b1_e1e-08_eta0.3_T*_s1.npz"
    files = sorted(ODE.glob(pat), key=lambda p: float(p.name.split("_T")[1].split("_s")[0]))
    return files[-1] if files else None


def tcs(t, L):
    return np.array([K.first_crossing(t, L[:, c], 0.5 * L[0, c]) for c in range(L.shape[1])])


def learned_at(t, L, T):
    k = np.searchsorted(t, T, side="right")
    return int(np.isfinite(tcs(t[:k], L[:k])).sum())


def main():
    out = {}
    print("| optimizer | ODE T | sim T | learned at t=316 sim / ODE | learned at sim T |"
          " t_c sim/ODE: median, max |log| (n) | stalled-class ell(T) sim / ODE (median) |")
    print("|---|---|---|---|---|---|---|")
    for opt in ("adam", "sgd"):
        g = K.find(opt, 128, 1.0, 1e-8 if opt == "adam" else None, alpha=1.0)
        if g is None:
            print(f"| {opt} | – | not run yet |")
            continue
        d = K.load(g["name"])
        for lane, b2 in enumerate(d["beta2s"]):
            f = ode_file(opt, float(b2))
            if f is None:
                continue
            z = np.load(f)
            ts, Ls = d["time"], d["loss"][:, lane].mean(1)           # (T, C) seed means
            to, Lo = z["time"], z["loss"]
            Tm = min(ts[-1], to[-1])
            ks, ko = ts <= Tm, to <= Tm
            a, b = tcs(ts[ks], Ls[ks]), tcs(to[ko], Lo[ko])
            ok = np.isfinite(a) & np.isfinite(b) & (b >= 0.1)
            lr = np.log(a[ok] / b[ok])
            stalled = ~np.isfinite(b)
            ell_s = np.interp(np.log(Tm), np.log(ts[1:]), np.log(Ls[1:].mean(-1)))
            rat_stall = (np.median(np.exp(np.array([np.interp(np.log(Tm), np.log(ts[1:]), np.log(Ls[1:, c]))
                                                     for c in np.flatnonzero(stalled)]))
                                    / np.exp(np.array([np.interp(np.log(Tm), np.log(to[1:]), np.log(Lo[1:, c]))
                                                       for c in np.flatnonzero(stalled)])))
                         if stalled.any() else np.nan)
            rec = {"ode_T": float(to[-1]), "sim_T": float(ts[-1]),
                   "learned316_sim": learned_at(ts, Ls, 316.3), "learned316_ode": learned_at(to, Lo, 316.3),
                   "learned_simT": int(np.isfinite(tcs(ts, Ls)).sum()),
                   "tc_median": float(np.exp(np.median(lr))) if ok.any() else np.nan,
                   "tc_maxabslog": float(np.abs(lr).max()) if ok.any() else np.nan, "n": int(ok.sum()),
                   "stalled_ratio": float(rat_stall)}
            out[f"{opt}_{b2:g}" if opt == "adam" else "sgd"] = rec
            lab = f"Adam {b2:g}" if opt == "adam" else "SGD"
            print(f"| {lab} | {rec['ode_T']:.0f} | {rec['sim_T']:.0f} | {rec['learned316_sim']} / {rec['learned316_ode']} |"
                  f" {rec['learned_simT']} | {rec['tc_median']:.3f}, {rec['tc_maxabslog']:.3f} ({rec['n']}) |"
                  f" {rec['stalled_ratio']:.3f} |")
    (K.OUTDIR / "s3b_gate.json").write_text(json.dumps(K.jsonable(out), indent=1))


if __name__ == "__main__":
    main()
