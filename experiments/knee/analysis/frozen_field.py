"""Series J Stage 4 (GPU part): the frozen ODE vector field evaluated on simulated states.

For every S3a group with ``eps = 1e-8`` (the psi grid, ``f0 = 1e-4``, cannot represent the
``eps = 1e-16`` floor) and every ``beta2`` lane, at the seed-mean states ``(B_mean, O_mean)``
on every other record with ``t >= 0.05``:

* Adam: psi-route ``phi, chi, sigma`` (``frozen.coefficients``, the series I plan);
  SGD: path-free moment coefficients;
* ``ito_split``: predicted ``d ell_c/dt`` = ``O`` part + ``B`` drift part + Ito part (P1, P6);
* ``balance``: the E4 / K2 balance ratio and ``mbar_pred`` (figure 6, P4);
* ``normaliser`` (Adam): own / background / eps regime and window / saturated label (figure 5);
* ``loss_state``: the MC class loss at the seed-mean state (to separate the field test from the
  Jensen gap between ``ell(mean state)`` and ``mean ell``).

Units: ``B``, ``O`` are invariant under the kappa rescaling; the field uses primed units
``r'^2 = 25/kappa``, ``eta'_A = sqrt(kappa) 0.3``, ``eta'_S = kappa 3``.

Output: ``figs/experiments/knee/analysis/field/<group>.npz``. Resumable (skips finished groups).
Run: ``real/bin/python experiments/knee/analysis/frozen_field.py [--groups substr ...]``.
"""
from __future__ import annotations

import argparse
import glob
import time
from pathlib import Path

import numpy as np

import frozen as Fz                      # sets x64, no preallocation, its own jax cache
import diagnostics as D

import jax
import jax.numpy as jnp
import discounts

REPO = Path(__file__).resolve().parents[3]
RUNS = REPO / "figs" / "experiments" / "knee" / "runs"
OUT = REPO / "figs" / "experiments" / "knee" / "analysis" / "field"
ETA = {"adam": 0.3, "sgd": 3.0}
CODES = {"eps": 0, "background": 1, "own": 2, "window": 0, "saturated": 1}


def parse(group):
    parts = group.split("_")
    opt, C = parts[1], int(parts[2][1:])
    alpha, kappa = float(parts[3][1:]), float(parts[4][1:])
    eps = float(parts[5][1:]) if opt == "adam" else 1e-8
    return opt, C, alpha, kappa, eps


def load_group(path):
    fs = sorted(glob.glob(str(path / "chunk_*.npz")))
    keys = ("time", "B_mean", "O_mean")
    data = {k: [] for k in keys}
    for f in fs:
        d = np.load(f)
        for k in keys:
            data[k].append(d[k])
        priors, beta2s = d["priors"], d["beta2s"]
    return {k: np.concatenate(v) for k, v in data.items()}, priors, beta2s


@jax.jit
def _sgd_coefficients(B, O, priors, draws):
    return discounts.softmax_moment_coefficients(B, O, priors, draws, 1e-8, 1.0, mode="sgd")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--groups", nargs="*", default=[], help="substrings to select groups")
    ap.add_argument("--every", type=int, default=2)
    ap.add_argument("--t-min", type=float, default=0.05)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    groups = sorted(p for p in RUNS.iterdir() if p.is_dir())
    groups = [p for p in groups if "_a0_" in p.name and ("e1e-16" not in p.name)]
    if args.groups:
        groups = [p for p in groups if any(s in p.name for s in args.groups)]
    # cheapest first
    groups.sort(key=lambda p: (parse(p.name)[1], parse(p.name)[0] == "adam"))
    for gp in groups:
        out = OUT / f"{gp.name}.npz"
        if out.exists():
            print(f"skip {gp.name} (done)", flush=True)
            continue
        opt, C, alpha, kappa, eps = parse(gp.name)
        data, priors, beta2s = load_group(gp)
        t = data["time"]
        idx = np.flatnonzero(t >= args.t_min)[::args.every]
        r2 = D.r_primed(kappa) ** 2
        eta = D.eta_primed(opt, kappa, ETA[opt])
        nl = len(beta2s)
        shape = (len(idx), nl, C)
        res = {k: np.full(shape, np.nan) for k in
               ("part_O", "part_B", "part_ito", "total", "loss_state", "ratio", "mbar",
                "mbar_pred", "margin_rate", "lambda", "forcing", "phi_cc", "m", "s2",
                "own", "bg", "R", "norm_code", "state_code", "rate_window", "rate_sat")}
        sgd_draws = jnp.asarray(np.random.default_rng(7).standard_normal((2 ** 15, C)))
        t0 = time.time()
        for j, k in enumerate(idx):
            for lane in range(nl):
                B, O = data["B_mean"][k, lane], data["O_mean"][k, lane]
                if opt == "adam":
                    co = Fz.coefficients(B, O, priors, float(beta2s[lane]), eps=eps, verbose=False)
                else:
                    c = _sgd_coefficients(jnp.asarray(B), jnp.asarray(O), jnp.asarray(priors),
                                          sgd_draws)
                    co = {q: np.asarray(c[q]) for q in ("phi", "chi", "sigma")}
                mom = D.class_moments(B, O, hessian=True)
                sp = D.ito_split(B, O, co["phi"], co["chi"], co["sigma"], r2=r2, eta=eta, mom=mom)
                bal = D.balance(O, co["phi"], co["chi"], r2)
                geo = D.geometry(B, O, np.sqrt(r2))
                res["part_O"][j, lane], res["part_B"][j, lane] = sp["O"], sp["B_drift"]
                res["part_ito"][j, lane], res["total"][j, lane] = sp["ito"], sp["total"]
                res["loss_state"][j, lane] = mom["loss"]
                for q in ("ratio", "mbar", "mbar_pred", "margin_rate", "lambda", "forcing",
                          "phi_cc"):
                    res[q][j, lane] = bal[q]
                res["m"][j, lane], res["s2"][j, lane] = geo["m"], geo["s2"]
                if opt == "adam":
                    nm = D.normaliser(mom, priors, beta2=float(beta2s[lane]), eps=eps)
                    res["own"][j, lane], res["bg"][j, lane], res["R"][j, lane] = (
                        nm["own"], nm["bg"], nm["R"])
                    res["norm_code"][j, lane] = np.vectorize(CODES.get)(nm["norm_regime"])
                    res["state_code"][j, lane] = np.vectorize(CODES.get)(nm["state_regime"])
                    res["rate_window"][j, lane] = nm["rate_window"]
                    res["rate_sat"][j, lane] = nm["rate_sat"]
            if j % 10 == 0:
                print(f"{gp.name}: {j + 1}/{len(idx)} states, t={t[k]:.3g}, "
                      f"{time.time() - t0:.0f}s", flush=True)
        np.savez_compressed(out, time=t[idx], record=idx, beta2s=beta2s, priors=priors,
                            r2=r2, eta=eta, kappa=kappa, eps=eps, opt=opt, C=C,
                            seconds=time.time() - t0, **res)
        print(f"done {gp.name}: {len(idx)} states x {nl} lanes, {time.time() - t0:.0f}s",
              flush=True)


if __name__ == "__main__":
    main()
