#!/usr/bin/env python3
"""Population cross-entropy optimum of the no-bias softmax on the Zipf mixture (series J, Stage 1).

Primed units (``K = Id``, means ``r' e_c``, ``r' = 5/sqrt(kappa)``). Write ``Theta'`` as
``Q W^T + Theta_perp`` with ``Q`` the orthonormal means. The logits are
``q = W (r' e_c + z) + n``, ``z ~ N(0, I_C)``, ``n = Theta_perp^T z_perp`` independent of the
class and of ``z``, so ``O = r' W`` and ``B = W W^T + N``. ``F(q; c) = lse(q) - q_c`` is convex
in ``q`` and ``E n = 0``, so by Jensen ``E_n F(q + n; c) >= F(q; c)`` for every ``q``: the risk
is minimised with ``Theta_perp = 0``, i.e. over ``W`` in ``R^{C x C}`` with

    R(W) = sum_c pi_c E_z F(r' W e_c + W z; c),       B = W W^T, O = r' W.

``R`` is convex in ``W`` and invariant under the shift gauge ``W -> W + 1 v^T`` (a common
shift of all logits); we fix ``1^T W = 0``.

* ``alpha = 0``: by permutation symmetry and convexity an optimum is ``W = a I``; the loss
  is the exact shape function ``L_C(m = a r', s^2 = 2 a^2)`` (``diagnostics.shape_loss``,
  a Frullani integral, no Monte Carlo). The model is then well specified (the Bayes
  posterior has logits ``r' x_k + const``), so ``a* = r'`` and ``R* = H(class | x)``; the
  numerical minimiser is checked against this.
* ``alpha > 0``: L-BFGS (scipy, jax gradients, float64) on

  - ``mc``: the exact risk with common random numbers (``n`` draws of ``z`` shared by all
    classes), full ``W`` (``C^2`` entries, gauge removed) or the ansatz
    ``W = diag(a) + b 1^T`` (row constant ``b``, ``sum b = 0``; ``b_k sum_j x_j`` acts as an
    effective bias ``b_k r'`` plus class-independent noise). Reported in sample and on
    fresh draws (out of sample).
  - ``pair``: ``R_pair(W) = sum_c pi_c sum_{k != c} E softplus(-(q_c - q_k))``, a
    deterministic 1-D Gaussian quadrature per pair. ``log(1 + sum x_k) <= sum log(1 + x_k)``,
    so ``R <= R_pair`` everywhere and ``min R_pair`` is a rigorous upper bound on ``R*``;
    it is the only usable objective when the optimum loss is far below MC resolution
    (``kappa <= 0.25``: a finite MC sample is then separable and its empirical optimum
    lies at infinity).

Usage: ``real/bin/python experiments/knee/analysis/optimum.py [--quick]``.
Output: ``figs/experiments/knee/analysis/optimum.json``.
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
import diagnostics as D  # noqa: E402  (enables x64)

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
from scipy.optimize import minimize, minimize_scalar  # noqa: E402

REPO = HERE.parents[2]
OUT = REPO / "figs" / "experiments" / "knee" / "analysis"
KAPPAS = (2.0, 1.0, 0.5, 0.25, 0.1)


def zipf_priors(C, alpha):
    pi = np.arange(1, C + 1, dtype=float) ** -alpha
    return pi / pi.sum()


def summary_classes(C):
    return {"head": 0, "median": C // 2 - 1, "rarest": C - 1}


# --------------------------------------------------------------------------- #
# alpha = 0: exact 1-D problem
# --------------------------------------------------------------------------- #
def symmetric_loss(C, r, a):
    return float(D.shape_loss_sigma(C, np.array([a * r]), a)[0])


def optimum_alpha0(C, kappa):
    r = D.r_primed(kappa)
    f = lambda la: np.log(symmetric_loss(C, r, np.exp(la)))
    sol = minimize_scalar(f, bracket=(np.log(0.7 * r), np.log(r), np.log(1.4 * r)),
                          tol=1e-7)
    a = float(np.exp(sol.x))
    ell = float(np.exp(sol.fun))
    scan = {f"{x:g}": symmetric_loss(C, r, x * r) for x in (0.9, 1.0, 1.1)}
    return {"a": a, "a_over_r": a / r, "R": ell, "ell": [ell] * C, "m": [a * r] * C,
            "s2": [2 * a * a] * C, "scan_a_over_r": scan, "nfev": int(sol.nfev)}


# --------------------------------------------------------------------------- #
# Objectives (jax)
# --------------------------------------------------------------------------- #
_U = jnp.arange(-45.0, 12.0 + 1e-9, 0.02)
_LOGW_U = -_U ** 2 / 2 - 0.5 * np.log(2 * np.pi) + np.log(0.02)


def _log_softplus(x):
    """``log(softplus(x))``, stable for very negative ``x`` and safe under ``where`` grads."""
    xs = jnp.maximum(x, -30.0)
    return jnp.where(x < -30.0, x, jnp.log(jax.nn.softplus(xs)))


def pair_matrix(W, r):
    """``E softplus(-(q_c - q_k) | c)`` for all ``(c, k)`` (diagonal = 0)."""
    C = W.shape[0]
    mu = r * (jnp.diag(W)[None, :] - W).T            # mu[c, k] = r (W_cc - W_kc)
    G = W @ W.T
    v = jnp.diag(G)[:, None] + jnp.diag(G)[None, :] - 2 * G
    s = jnp.sqrt(jnp.maximum(v, 1e-30))
    off = ~jnp.eye(C, dtype=bool)
    d = mu[..., None] + s[..., None] * _U
    val = jnp.exp(jax.nn.logsumexp(_log_softplus(-d) + _LOGW_U, axis=-1))
    return jnp.where(off, val, 0.0)


def pair_risk(W, r, priors):
    return jnp.sum(priors * pair_matrix(W, r).sum(1))


def mc_class_losses(W, r, Z):
    """Exact per-class losses by MC with common draws ``Z`` (``(n, C)``) for all classes."""
    C = W.shape[0]
    Q0 = Z @ W.T

    def one(c):
        q = Q0 + r * W[:, c][None, :]
        own = jax.nn.one_hot(c, C, dtype=bool)
        a = jax.nn.logsumexp(jnp.where(own[None, :], -jnp.inf, q), axis=1) - q[:, c]
        return jnp.logaddexp(0.0, a).mean()

    return jax.lax.map(one, jnp.arange(C))


def ansatz_W(params, C):
    a, b = params[:C], params[C:]
    return jnp.diag(a) + (b - b.mean())[:, None] * jnp.ones((1, C))


def ansatz_class_losses(params, r, Z):
    """Same as :func:`mc_class_losses` for ``W = diag(a) + b 1^T`` in ``O(n C)``:
    logit ``k`` on class ``c`` is ``a_k z_k + b_k (r' + sum_j z_j) + r' a_c [k = c]``."""
    C = Z.shape[1]
    a, b = params[:C], params[C:]
    b = b - b.mean()
    base = Z * a[None, :] + (r + Z.sum(1))[:, None] * b[None, :]
    ninf = jnp.full((Z.shape[0], 1), -jnp.inf)
    pre = jnp.concatenate([ninf, jax.lax.cumlogsumexp(base, axis=1)[:, :-1]], 1)
    suf = jnp.concatenate([jax.lax.cumlogsumexp(base, axis=1, reverse=True)[:, 1:], ninf], 1)
    loo = jnp.logaddexp(pre, suf)                                  # lse_{k != c} base_k
    return jnp.logaddexp(0.0, loo - base - r * a[None, :]).mean(0)


def gauge_W(V):
    return V - V.mean(0, keepdims=True)


def _lbfgs(fun, x0, maxiter=3000, gtol=1e-10):
    vg = jax.jit(jax.value_and_grad(fun))

    def f(x):
        v, g = vg(jnp.asarray(x))
        return float(v), np.asarray(g, dtype=float)

    t0 = time.time()
    sol = minimize(f, np.asarray(x0, dtype=float), jac=True, method="L-BFGS-B",
                   options={"maxiter": maxiter, "maxfun": 2 * maxiter, "gtol": gtol,
                            "ftol": 1e-15})
    return sol, time.time() - t0


def draws(C, n, seed):
    g = np.random.default_rng(seed + 7 * C).standard_normal((n // 2, C))
    return jnp.asarray(np.concatenate([g, -g]))


# --------------------------------------------------------------------------- #
# alpha > 0 solvers
# --------------------------------------------------------------------------- #
def solve_pair(C, r, priors, *, full, x0=None):
    pr = jnp.asarray(priors)
    if full:
        fun = lambda v: jnp.log(pair_risk(gauge_W(v.reshape(C, C)), r, pr))
        x0 = np.eye(C).ravel() * r if x0 is None else x0
        sol, sec = _lbfgs(fun, x0)
        W = np.asarray(gauge_W(jnp.asarray(sol.x).reshape(C, C)))
    else:
        fun = lambda p: jnp.log(pair_risk(ansatz_W(p, C), r, pr))
        x0 = np.concatenate([np.full(C, r), np.zeros(C)]) if x0 is None else x0
        sol, sec = _lbfgs(fun, x0)
        W = np.asarray(ansatz_W(jnp.asarray(sol.x), C))
    ell_pair = np.asarray(pair_matrix(jnp.asarray(W), r).sum(1))
    return {"W": W, "x": sol.x, "R_pair": float(np.exp(sol.fun)), "ell_pair": ell_pair,
            "nit": int(sol.nit), "success": bool(sol.success), "msg": str(sol.message),
            "seconds": sec}


def solve_mc(C, r, priors, *, full, n, seed, x0=None):
    pr = jnp.asarray(priors)
    Z = draws(C, n, seed)
    if full:
        fun = lambda v: jnp.log(jnp.sum(pr * mc_class_losses(gauge_W(v.reshape(C, C)), r, Z)))
        x0 = (np.eye(C) * r).ravel() if x0 is None else x0
        sol, sec = _lbfgs(fun, x0)
        W = np.asarray(gauge_W(jnp.asarray(sol.x).reshape(C, C)))
    else:
        fun = lambda p: jnp.log(jnp.sum(pr * ansatz_class_losses(p, r, Z)))
        x0 = np.concatenate([np.full(C, r), np.zeros(C)]) if x0 is None else x0
        sol, sec = _lbfgs(fun, x0)
        W = np.asarray(ansatz_W(jnp.asarray(sol.x), C))
    return {"W": W, "x": sol.x, "R_in": float(np.exp(sol.fun)), "nit": int(sol.nit),
            "success": bool(sol.success), "msg": str(sol.message), "seconds": sec,
            "scale": float(np.linalg.norm(W) / np.sqrt(C))}


def evaluate_mc(W, r, priors, *, n, seed, chunks=4):
    """Out-of-sample per-class losses (``chunks`` independent draw sets; mean and s.e.)."""
    C = W.shape[0]
    f = jax.jit(lambda w, z: mc_class_losses(w, r, z))
    vals = np.array([np.asarray(f(jnp.asarray(W), draws(C, n, seed + 1000 * (j + 1))))
                     for j in range(chunks)])
    ell = vals.mean(0)
    se = vals.std(0, ddof=1) / np.sqrt(chunks)
    Rv = vals @ priors
    return ell, se, float(Rv.mean()), float(Rv.std(ddof=1) / np.sqrt(chunks))


def ansatz_residual(W):
    """Relative Frobenius norm of the part of ``P W`` not of the form diag + row constant
    (+ gauge): least squares projection onto ``{diag(a) + b 1^T + 1 v^T}``."""
    C = W.shape[0]
    off = ~np.eye(C, dtype=bool)
    # off-diagonal entries should be b_k + v_j: two-way additive fit (alternating means)
    X = np.where(off, W, np.nan)
    b = np.zeros(C)
    v = np.zeros(C)
    for _ in range(200):
        b = np.nanmean(X - v[None, :], axis=1)
        v = np.nanmean(X - b[:, None], axis=0)
    res = np.where(off, X - b[:, None] - v[None, :], 0.0)
    return float(np.linalg.norm(res) / np.linalg.norm(W - W.mean(0, keepdims=True)))


def margins_of(W, r):
    m, m_min = D.margins(r * W)
    return m, m_min


def pack(C, priors, W, ell, r, extra):
    m, m_min = margins_of(W, r)
    ell = np.asarray(ell)
    sel = summary_classes(C)
    out = {"R": float(ell @ priors), "ell": ell.tolist(), "ell_rel": (ell / np.log(C)).tolist(),
           "m": m.tolist(), "m_min": m_min.tolist(),
           "s2": D.diff_var(W @ W.T).tolist(),
           "summary": {k: {"pi": float(priors[c]), "ell": float(ell[c]),
                           "ell_rel": float(ell[c] / np.log(C)), "m": float(m[c])}
                       for k, c in sel.items()},
           "diagW": np.diag(W).tolist(), "rowconst": (W - np.diag(np.diag(W))).sum(1).tolist()}
    out.update(extra)
    return out


# --------------------------------------------------------------------------- #
# Drivers
# --------------------------------------------------------------------------- #
def run_alpha0(C, kappa, results):
    key = f"C{C}_a0_k{kappa:g}"
    t0 = time.time()
    r = D.r_primed(kappa)
    ex = optimum_alpha0(C, kappa)
    priors = np.full(C, 1.0 / C)
    W = ex["a"] * np.eye(C)
    pair = float(np.asarray(pair_risk(jnp.asarray(W), r, jnp.asarray(priors))))
    rec = pack(C, priors, W, ex["ell"], r, {
        "method": "exact (Frullani 1-D, W = a I)", "r_prime": r, "a": ex["a"],
        "a_over_r": ex["a_over_r"], "bayes_check": "a* = r' expected (well specified)",
        "scan_a_over_r": ex["scan_a_over_r"], "R_pair_at_opt": pair,
        "pair_over_exact": pair / ex["R"], "log10_R": float(np.log10(ex["R"])),
        "seconds": time.time() - t0})
    if kappa >= 0.5:
        ell, se, Rv, Rse = evaluate_mc(W, r, priors, n=2 ** 16, seed=11)
        rec["mc_check"] = {"R": Rv, "R_se": Rse}
    results[key] = rec
    print(f"{key}: a/r'={ex['a_over_r']:.5f} R*={ex['R']:.4e} (ell/logC={ex['R']/np.log(C):.3e})"
          f" m*={ex['a']*r:.2f} pair/exact={pair/ex['R']:.3f}"
          + (f" MC={rec['mc_check']['R']:.4e}+-{rec['mc_check']['R_se']:.1e}" if 'mc_check' in rec else "")
          + f" [{time.time()-t0:.0f}s]", flush=True)


def run_alpha(C, alpha, kappa, results, *, n_mc, validate=False):
    key = f"C{C}_a{alpha:g}_k{kappa:g}"
    r = D.r_primed(kappa)
    priors = zipf_priors(C, alpha)
    rec = {"r_prime": r, "kappa": kappa}
    t0 = time.time()
    pa = solve_pair(C, r, priors, full=False)
    pf = solve_pair(C, r, priors, full=True,
                    x0=np.asarray(pa["W"]).ravel())
    rec["pair_ansatz"] = {"R_pair": pa["R_pair"], "nit": pa["nit"], "success": pa["success"],
                          "seconds": pa["seconds"]}
    rec["pair_full"] = {"R_pair": pf["R_pair"], "nit": pf["nit"], "success": pf["success"],
                        "seconds": pf["seconds"], "ansatz_residual": ansatz_residual(pf["W"]),
                        "full_over_ansatz": pf["R_pair"] / pa["R_pair"]}
    best = {"W": pf["W"], "ell": pf["ell_pair"], "method": "pair upper bound (full W)"}
    if kappa >= 0.5:
        # exact risk of the pair-optimal W on fresh draws (an honest upper bound on R*;
        # at kappa = 0.5 the MC is dominated by rare events and is only a rough check)
        ell_p, se_p, R_p, Rse_p = evaluate_mc(pf["W"], r, priors, n=n_mc, seed=1)
        rec["exact_at_pair_opt"] = {"R": R_p, "R_se": Rse_p, "pair_over_exact": pf["R_pair"] / R_p}
    if kappa >= 1.0:
        ma = solve_mc(C, r, priors, full=False, n=n_mc, seed=1, x0=pa["x"])
        ell_a, se_a, R_a, Rse_a = evaluate_mc(ma["W"], r, priors, n=n_mc, seed=1)
        rec["mc_ansatz"] = {"R_in": ma["R_in"], "R_out": R_a, "R_out_se": Rse_a,
                            "optimism": ma["R_in"] / R_a - 1,
                            "nit": ma["nit"], "success": ma["success"],
                            "seconds": ma["seconds"], "ell_se_rel_max": float(np.max(se_a / ell_a))}
        cands = [(R_a, ma["W"], ell_a, "MC ansatz (out of sample)"),
                 (R_p, pf["W"], ell_p, "pair-optimal W, exact risk by MC (out of sample)")]
        R_best, W_best, ell_best, how = min(cands, key=lambda c: c[0])
        best = {"W": W_best, "ell": ell_best, "method": how}
        if validate:
            mf = solve_mc(C, r, priors, full=True, n=n_mc, seed=1, x0=np.asarray(ma["W"]).ravel())
            ell_f, se_f, R_f, Rse_f = evaluate_mc(mf["W"], r, priors, n=n_mc, seed=1)
            rec["mc_full"] = {"R_in": mf["R_in"], "R_out": R_f, "R_out_se": Rse_f,
                              "nit": mf["nit"], "success": mf["success"],
                              "seconds": mf["seconds"],
                              "ansatz_residual": ansatz_residual(mf["W"]),
                              "in_full_over_ansatz": mf["R_in"] / ma["R_in"],
                              "out_full_over_ansatz": R_f / R_a,
                              "expected_in_sample_gap": (C * C - 2 * C) / (2 * n_mc * C) / ma["R_in"]}
    if kappa <= 0.25:
        # does the MC empirical optimum run away? (separable sample)
        Z = draws(C, n_mc, 1)
        f = jax.jit(lambda p: jnp.sum(jnp.asarray(priors) * ansatz_class_losses(p, r, Z)))
        scales = (1.0, 2.0, 4.0, 8.0)
        vals = [float(f(jnp.asarray(pa["x"]) * s)) for s in scales]
        rec["mc_runaway"] = {"scales": scales, "R_emp": vals,
                             "monotone_decreasing": bool(np.all(np.diff(vals) < 0))}
    rec.update(pack(C, priors, best["W"], best["ell"], r, {"method": best["method"]}))
    rec["ell_pair_at_best"] = np.asarray(pair_matrix(jnp.asarray(best["W"]), r).sum(1)).tolist()
    rec["log10_R"] = float(np.log10(rec["R"]))
    rec["seconds"] = time.time() - t0
    results[key] = rec
    s = rec["summary"]
    msg = (f"{key}: {best['method']}: R*={rec['R']:.4e}; pair: ansatz {pa['R_pair']:.4e},"
           f" full {pf['R_pair']:.4e} (resid {rec['pair_full']['ansatz_residual']:.3f})")
    if "exact_at_pair_opt" in rec:
        e = rec["exact_at_pair_opt"]
        msg += f"; exact MC at pair-opt W {e['R']:.4e}+-{e['R_se']:.1e}"
    if "mc_ansatz" in rec:
        msg += (f"; MC ansatz in {rec['mc_ansatz']['R_in']:.4e} out {rec['mc_ansatz']['R_out']:.4e}"
                f"+-{rec['mc_ansatz']['R_out_se']:.1e}")
    if "mc_full" in rec:
        mf = rec["mc_full"]
        msg += (f"; MC full in {mf['R_in']:.4e} out {mf['R_out']:.4e} (resid {mf['ansatz_residual']:.3f},"
                f" expected in-sample gap {mf['expected_in_sample_gap']:.2%})")
    if "mc_runaway" in rec:
        msg += f"; MC empirical risk vs scale {['%.2e' % v for v in rec['mc_runaway']['R_emp']]}"
    msg += (f"\n    head ell/logC={s['head']['ell_rel']:.3e} m={s['head']['m']:.2f};"
            f" median {s['median']['ell_rel']:.3e} m={s['median']['m']:.2f};"
            f" rarest {s['rarest']['ell_rel']:.3e} m={s['rarest']['m']:.2f} [{rec['seconds']:.0f}s]")
    print(msg, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="C = 32 validation only")
    ap.add_argument("--skip-validate", action="store_true")
    ap.add_argument("--skip-alpha0", action="store_true")
    ap.add_argument("--n-mc", type=int, default=2 ** 17)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "optimum.json"
    results = json.loads(path.read_text()) if path.exists() else {}

    def save():
        path.write_text(json.dumps(results, indent=1))

    if not args.skip_alpha0:
        for C in (32, 128):
            for kappa in KAPPAS:
                run_alpha0(C, kappa, results)
                save()
    if not args.skip_validate:
        for alpha in (1.0, 2.0):
            for kappa in KAPPAS:
                run_alpha(32, alpha, kappa, results, n_mc=args.n_mc, validate=kappa >= 0.5)
                save()
    if args.quick:
        return
    for alpha in (1.0, 2.0):
        for kappa in KAPPAS:
            run_alpha(128, alpha, kappa, results, n_mc=args.n_mc)
            save()
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
