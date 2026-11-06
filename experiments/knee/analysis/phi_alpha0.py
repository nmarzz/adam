#!/usr/bin/env python3
"""phi_cc, chi_cc and the E4 balance margin at alpha = 0 (uniform priors), kappa = 1.

Frozen psi-route coefficients (``frozen.coefficients``: the series I plan, seed 0) at

* ``Theta_0``: ``B = 0.01 Id``, ``O = 0`` (the ODE start, ``kappa = 1``);
* synthetic learned states with margin exactly ``m``: ``O = m (I - 11^T/C)``,
  ``B = (m^2/r'^2)(I - 11^T/C) + nu I`` (signal part ``O O^T/r'^2``, so ``gamma = 2``,
  plus isotropic noise ``nu``; ``nu = 0.01`` is the start value, ``nu = 0.3`` is typical of
  ``t ~ 10`` on the cached ``C = 32`` trajectories, where ``s_noise^2/2`` is 0.31-0.40).

At ``alpha = 0`` every class is equivalent, so a joint fixed point of the gaps needs the
margin rate ``(r'^2|chi_cc| + lambda)(1 - ratio)`` (``diagnostics.balance``) to change sign
along the family: ``mbar_pred(m) = (r'^2|chi_cc| + lambda)/phi_cc`` crosses ``m``.

Usage: ``real/bin/python experiments/knee/analysis/phi_alpha0.py``;
writes ``figs/experiments/knee/analysis/phi_alpha0.json``.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import diagnostics as D  # noqa: E402
import frozen  # noqa: E402
import lib  # noqa: E402

OUT = HERE.parents[2] / "figs" / "experiments" / "knee" / "analysis"
MS = (1.0, 2.0, 3.0, 4.5, 6.0, 9.0, 12.0, 16.0, 20.0)
NUS = (0.01, 0.3)


def states(C, r):
    P = np.eye(C) - 1.0 / C
    yield "theta0", 0.0, None, 0.01 * np.eye(C), np.zeros((C, C))
    for nu in NUS:
        for m in MS:
            yield f"m{m:g}_nu{nu:g}", m, nu, (m * m / r ** 2) * P + nu * np.eye(C), m * P


def main():
    kappa = 1.0
    r = D.r_primed(kappa)
    r2 = r * r
    path = OUT / "phi_alpha0.json"
    res = json.loads(path.read_text()) if path.exists() else {}
    for C in (32, 128):
        priors = np.full(C, 1.0 / C)
        for beta2 in (0.9, 0.95, 0.99, 0.999):
            key = f"C{C}_b{beta2:g}"
            if key in res:
                continue
            t0 = time.time()
            rows = {}
            for name, m, nu, B, O in states(C, r):
                co = frozen.coefficients(B, O, priors, beta2, verbose=False)
                bal = D.balance(O, co["phi"], co["chi"], r2)
                mom = D.class_moments(B, O)
                nm = D.normaliser(mom, priors, beta2=beta2)
                mr = bal["margin_rate"]
                rows[name] = {
                    "m": m, "nu": nu,
                    "phi_cc": float(bal["phi_cc"].mean()), "phi_cc_cv": float(bal["phi_cc"].std() / bal["phi_cc"].mean()),
                    "chi_cc": float(bal["chi_cc"].mean()), "chi_cc_over_pi": float(bal["chi_cc"].mean() * C),
                    "lambda": float(bal["lambda"].mean()), "forcing": float(bal["forcing"].mean()),
                    "mbar_pred": float(bal["mbar_pred"].mean()), "mbar_pred_sd": float(bal["mbar_pred"].std()),
                    "margin_rate": float(mr.mean()), "margin_rate_sd": float(mr.std()),
                    "ratio": float(bal["ratio"].mean()) if m > 0 else None,
                    "loss": float(mom["loss"].mean()), "g1": float(mom["g1"].mean()),
                    "own": float(nm["own"].mean()), "bg": float(nm["bg"].mean()), "z": float(nm["z"].mean()),
                    "tau": float(nm["tau"].mean()), "rate_window": float(nm["rate_window"].mean()),
                    "rate_sat": float(nm["rate_sat"].mean()), "state_regime": str(nm["state_regime"][0]),
                    "norm_regime": str(nm["norm_regime"][0]),
                    "psi_sat_F": float(D.psi_sat(beta2) * lib.F_pred(beta2, C)),
                    "kink_over_pi": float(lib.kink(beta2, C) * C)}
            # fixed point: sign change of the margin rate along m (per nu)
            fp = {}
            for nu in NUS:
                ms = np.array(MS)
                rate = np.array([rows[f"m{m:g}_nu{nu:g}"]["margin_rate"] for m in MS])
                gap = np.array([rows[f"m{m:g}_nu{nu:g}"]["mbar_pred"] for m in MS]) - ms
                k = np.flatnonzero(np.sign(rate[:-1]) != np.sign(rate[1:]))
                if len(k):
                    i = k[0]
                    w = rate[i] / (rate[i] - rate[i + 1])
                    fp[f"nu{nu:g}"] = {"crossing_m": float(ms[i] + w * (ms[i + 1] - ms[i])),
                                       "rates": rate.tolist(), "mbar_minus_m": gap.tolist()}
                else:
                    fp[f"nu{nu:g}"] = {"crossing_m": None, "rates": rate.tolist(),
                                       "mbar_minus_m": gap.tolist()}
            res[key] = {"states": rows, "fixed_point": fp, "seconds": time.time() - t0,
                        "regime_kink": ("frequent (1/C above kink)" if 1.0 / C > lib.kink(beta2, C)
                                        else "rare (1/C below kink)"),
                        "oneoverC_over_kink": float(1.0 / C / lib.kink(beta2, C))}
            path.write_text(json.dumps(res, indent=1))
            th = rows["theta0"]
            print(f"C={C} beta2={beta2:g} [{time.time() - t0:.0f}s] 1/C / kink = {res[key]['oneoverC_over_kink']:.2f}:"
                  f" Theta0 phi_cc={th['phi_cc']:.4g} chi_cc/pi={th['chi_cc_over_pi']:.4g}"
                  f" (psi_sat F={th['psi_sat_F']:.4g}, window {th['rate_window']:.4g}) lambda={th['lambda']:.3g}"
                  f" mbar_pred={th['mbar_pred']:.3g}", flush=True)
            for nu in NUS:
                print(f"   nu={nu:g}: m -> (phi_cc, chi_cc/pi, lambda/forcing, mbar_pred, margin rate):", flush=True)
                for m in MS:
                    q = rows[f"m{m:g}_nu{nu:g}"]
                    print(f"      m={m:g}: phi={q['phi_cc']:.4g} chi/pi={q['chi_cc_over_pi']:.4g}"
                          f" lam/F={q['lambda']/q['forcing']:.3g} mbar_pred={q['mbar_pred']:.3g}"
                          f" (sd {q['mbar_pred_sd']:.2g}) rate={q['margin_rate']:.4g} [{q['state_regime']},"
                          f" {q['norm_regime']}, z={q['z']:.3g}, ell={q['loss']:.3g}]", flush=True)
                print(f"      fixed point crossing: {fp[f'nu{nu:g}']['crossing_m']}", flush=True)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
