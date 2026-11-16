"""Single-Gaussian Adam ODE vs Adam on real data (scalar linear regression).

Population = empirical distribution of a processed dataset (realdata_prep.py):

    K = X^T X / n,  theta* = K^{-1} X^T y / n,  omega = y - X theta*,
    R(theta) = 1/2 (theta - theta*)^T K (theta - theta*) + sigma^2 / 2   (exact).

Adam streams iid rows of the dataset (sampling with replacement, i.e. exactly
the theory's one-pass setting with the empirical law as data distribution);
the only departures from the theory are non-Gaussian x and omega's dependence
on x.  A "Gaussian twin" (x ~ N(0, K), y = x theta* + omega with omega drawn
independently from the residuals or from N(0, sigma^2)) isolates finite-D
effects from non-Gaussian ones.

The ODE (appendix eq. system_ODEs, M = M* = 1) in per-mode errors
e_i = (theta - theta*)^T K Pi_i (theta - theta*) reads

    de_i/dt = -2 eta lambda_i a(P) e_i
              + (eta^2 / D) lambda_i w_i^T Dh g(C; P) Dh w_i,

with P = sum_i e_i / 2, lambda_i, w_i the eigenpairs of the correlation-type
matrix diag(K)^{-1/4} K diag(K)^{-1/4}, Dh = diag(K)^{1/4}, C the correlation
matrix, and g(rho; P) = E[A^j A^k] the Sigma entry for two coordinates with
correlation rho (applied entrywise).  a and g are tabulated once by simulating
long stationary single-coordinate Adam chains (no history truncation).

Commands (repository root):
    real/bin/python experiments/realdata.py diag  NAME
    real/bin/python experiments/realdata.py run   NAME --eta 1 --beta1 0.9 --beta2 0.999 --T 20
"""
from __future__ import annotations

import argparse
import json
import math
import time
from functools import partial
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from jax import lax  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"
RES = ROOT / "results" / "realdata"
FIGS = ROOT / "figs" / "experiments" / "realdata"

RHO_NODES = np.array([0.0, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7,
                      0.8, 0.9, 0.95, 0.98, 1.0])


# --------------------------------------------------------------------------- #
# Population statistics
# --------------------------------------------------------------------------- #
def load(name):
    z = np.load(DATA / f"{name}.npz")
    return z["X"], z["y"], str(z["info"])


def population(X, y, chunk=131_072):
    n, D = X.shape
    K = jnp.zeros((D, D))
    b = jnp.zeros(D)
    for a in range(0, n, chunk):
        Xc = jnp.asarray(X[a:a + chunk], jnp.float64)
        K = K + Xc.T @ Xc
        b = b + Xc.T @ jnp.asarray(y[a:a + chunk], jnp.float64)
    K, b = K / n, b / n
    ridge = 1e-12 * jnp.trace(K) / D
    theta = jnp.linalg.solve(K + ridge * jnp.eye(D), b)
    resid = np.concatenate([
        np.asarray(jnp.asarray(y[a:a + chunk], jnp.float64)
                   - jnp.asarray(X[a:a + chunk], jnp.float64) @ theta)
        for a in range(0, n, chunk)])
    return K, theta, resid


def spectral(K):
    d4 = jnp.diag(K) ** 0.25
    lam, W = jnp.linalg.eigh(K / d4[:, None] / d4[None, :])
    lam = jnp.clip(lam, 0.0)
    sd = jnp.sqrt(jnp.diag(K))
    C = K / sd[:, None] / sd[None, :]
    return lam, W, d4, C


def kurt(v, axis=0):
    v = v - v.mean(axis=axis, keepdims=True)
    return (v ** 4).mean(axis=axis) / (v ** 2).mean(axis=axis) ** 2 - 3.0


def diagnostics(name, X, y, K, theta, resid, lam, W, d4, C, nsub=200_000, seed=0):
    rng = np.random.default_rng(seed)
    n, D = X.shape
    idx = rng.choice(n, min(n, nsub), replace=False)
    Xs = jnp.asarray(X[idx], jnp.float64)
    sd = jnp.sqrt(jnp.diag(K))
    Xi = Xs / sd
    w = jnp.asarray(resid[idx])
    sig2 = float(np.mean(resid ** 2))
    kx = np.asarray(kurt(Xi))                          # coordinate marginals
    proj_star = Xs @ theta
    G = jnp.asarray(rng.standard_normal((D, 8)))
    proj_rand = Xs @ G
    # Adam denominator factorisation at the optimum: E[w^2 xi_j^2] / E[w^2]
    fact = np.asarray((w[:, None] ** 2 * Xi ** 2).mean(0) / (w ** 2).mean())
    norms2 = jnp.sum(Xs ** 2, axis=1)
    e0 = np.asarray(lam * (W.T @ (d4 * theta)) ** 2)  # initial per-mode error
    lam_np = np.asarray(lam)
    off = np.asarray(jnp.abs(C - jnp.diag(jnp.diag(C))))
    out = dict(
        name=name, n=int(n), D=int(D),
        R0=float(0.5 * np.mean(y.astype(np.float64) ** 2)), sigma2=sig2,
        r2=float(1 - sig2 / np.var(y.astype(np.float64))),
        diagK_spread=float(np.percentile(np.asarray(jnp.diag(K)), 95)
                           / np.percentile(np.asarray(jnp.diag(K)), 5)),
        eff_dim_pr=float(lam_np.sum() ** 2 / (lam_np ** 2).sum()),
        eff_dim_trop=float(lam_np.sum() / lam_np.max()),
        signal_eff_modes=float(e0.sum() ** 2 / (e0 ** 2).sum()),
        signal_top10_frac=float(np.sort(e0)[::-1][:10].sum() / e0.sum()),
        mean_abs_offdiag_C=float(off.sum() / (D * (D - 1))),
        coord_exkurt_median=float(np.median(kx)),
        coord_exkurt_p90=float(np.percentile(kx, 90)),
        coord_exkurt_max=float(kx.max()),
        proj_star_exkurt=float(kurt(proj_star)),
        proj_rand_exkurt=float(np.mean(np.asarray(kurt(proj_rand)))),
        resid_exkurt=float(kurt(jnp.asarray(resid))),
        resid_skew=float(np.mean((resid - resid.mean()) ** 3) / sig2 ** 1.5),
        hetero_corr_w2_norm2=float(np.corrcoef(np.asarray(w) ** 2, np.asarray(norms2))[0, 1]),
        denom_factor_median=float(np.median(fact)),
        denom_factor_p10=float(np.percentile(fact, 10)),
        denom_factor_p90=float(np.percentile(fact, 90)),
    )
    return out


# --------------------------------------------------------------------------- #
# Stationary coefficients a(P), g(rho; P) from single-coordinate Adam chains
# --------------------------------------------------------------------------- #
@partial(jax.jit, static_argnames=("n", "burn", "keep", "tail", "gauss"))
def _chain(key, P, sigma, resid, beta1, beta2, eps, rhos, *, n, burn, keep, tail, gauss):
    """Returns (a_deriv, a_ibp, g[rhos]) for one value of P.

    f = s - omega with s ~ N(0, 2P), omega ~ N(0, sigma^2) (gauss) or a
    bootstrap draw from ``resid``.  Coordinate j sees xi = z1, coordinate k
    sees rho z1 + sqrt(1 - rho^2) z2 (one column per rho node, rho = 1 last).
    """
    r = rhos.shape[0]
    c = jnp.sqrt(1 - rhos ** 2)

    def draw(k):
        k1, k2, k3, k4 = jax.random.split(k, 4)
        s = jnp.sqrt(2 * P) * jax.random.normal(k1, (n,), jnp.float32)
        if gauss:
            om = sigma * jax.random.normal(k2, (n,), jnp.float32)
        else:
            om = resid[jax.random.randint(k2, (n,), 0, resid.shape[0])]
        z1 = jax.random.normal(k3, (n,), jnp.float32)
        z2 = jax.random.normal(k4, (n,), jnp.float32)
        xi = rhos[None, :] * z1[:, None] + c[None, :] * z2[:, None]   # (n, r)
        return s, s - om, xi

    def fwd(carry, k):
        V, N1, N2, Nibp = carry
        s, f, xi = draw(k)
        V = beta2 * V + (1 - beta2) * (f[:, None] * xi) ** 2
        x1 = xi[:, -1]
        N1 = beta1 * N1 + (1 - beta1) * x1 ** 2
        N2 = beta1 * beta2 * N2 + (1 - beta1) * (1 - beta2) * x1 ** 4 * f ** 2
        Nibp = beta1 * Nibp + (1 - beta1) * s * f * x1 ** 2
        return (V, N1, N2, Nibp), (f, xi, V)

    z = jnp.zeros((n,), jnp.float32)
    carry0 = (jnp.zeros((n, r), jnp.float32), z, z, z)
    keys = jax.random.split(key, burn + keep + tail)
    carry, _ = lax.scan(lambda cr, k: (fwd(cr, k)[0], None), carry0, keys[:burn])

    def fwd_rec(cr, k):
        cr, (f, xi, V) = fwd(cr, k)
        Vj = jnp.maximum(cr[0][:, -1] + eps, 1e-30)
        a_d = cr[1] / jnp.sqrt(Vj) - cr[2] / Vj ** 1.5
        a_i = cr[3] / jnp.sqrt(Vj)
        return cr, (f, xi, lax.rsqrt(jnp.maximum(V + eps, 1e-30)), a_d.mean(dtype=jnp.float64),
                    a_i.mean(dtype=jnp.float64))

    _, (f, xi, isq, ad, ai) = lax.scan(fwd_rec, carry, keys[burn:])
    # R_t = sum_l beta1^l isq_{t+l}  (backward recursion), then A_t = (1-b1) f xi R.
    def back(Rn, x):
        R = x + beta1 * Rn
        return R, R
    _, R = lax.scan(back, jnp.zeros_like(isq[0]), isq, reverse=True)
    A = (1 - beta1) * f[:keep, :, None] * xi[:keep] * R[:keep]        # (keep, n, r)
    g = jnp.einsum("tn,tnr->r", A[..., -1].astype(jnp.float64),
                   A.astype(jnp.float64)) / (keep * n)
    return ad[:keep].mean() / 1.0, ai[:keep].mean() / (2 * P), g


def chain_coefficients(P_grid, beta1, beta2, eps, resid, *, gauss, n=4096, keep=4000,
                       seed=0):
    sigma = float(np.sqrt(np.mean(resid ** 2)))
    burn = int(math.ceil(12 / (1 - beta2)))
    tail = 1 if beta1 == 0 else int(math.ceil(math.log(1e-7) / math.log(beta1)))
    rhos = jnp.asarray(RHO_NODES, jnp.float32)
    r_dev = jnp.asarray(resid, jnp.float32)
    a_d, a_i, g = [], [], []
    key = jax.random.PRNGKey(seed)
    for P in P_grid:
        key, sub = jax.random.split(key)
        out = _chain(sub, jnp.float32(P), jnp.float32(sigma), r_dev, jnp.float32(beta1),
                     jnp.float32(beta2), jnp.float32(eps), rhos,
                     n=n, burn=burn, keep=keep, tail=tail, gauss=gauss)
        a_d.append(float(out[0])); a_i.append(float(out[1])); g.append(np.asarray(out[2]))
    return np.array(a_d), np.array(a_i), np.array(g)


# --------------------------------------------------------------------------- #
# ODE
# --------------------------------------------------------------------------- #
def odd_hats(C, nodes):
    """Basis o_k(C) with g(C) = sum_k g(nodes[k]) o_k(C) for odd, piecewise-linear g."""
    A = jnp.abs(C)
    sgn = jnp.sign(C)
    hats = []
    for k in range(1, len(nodes)):
        lo = nodes[k - 1]
        hi = nodes[k + 1] if k + 1 < len(nodes) else None
        up = jnp.clip((A - lo) / (nodes[k] - lo), 0, 1)
        down = 1.0 if hi is None else jnp.clip((hi - A) / (hi - nodes[k]), 0, 1)
        hats.append(sgn * jnp.minimum(up, down) * (A >= lo))
    return hats  # node 0 (rho = 0) has g = 0 by oddness


def gamma_table(lam, W, d4, C):
    """Gamma[k, i] = lambda_i w_i^T Dh o_k(C) Dh w_i / D for nodes 1..."""
    D = len(lam)
    DW = d4[:, None] * W
    rows = []
    for o in odd_hats(C, RHO_NODES):
        rows.append(lam * jnp.sum(DW * (o @ DW), axis=0) / D)
    return jnp.stack(rows)


@partial(jax.jit, static_argnames=("steps", "every"))
def _ode(e0, lam, Gam, logP, a_tab, g_tab, eta, T, *, steps, every):
    dt = T / steps

    def interp(P):
        lp = jnp.log(jnp.maximum(P, 1e-300))
        a = jnp.exp(jnp.interp(lp, logP, jnp.log(a_tab)))
        g = jax.vmap(lambda col: jnp.interp(lp, logP, col), in_axes=1)(g_tab)
        return a, g

    def step(e, _):
        P = 0.5 * e.sum()
        a, g = interp(P)
        rate = 2 * eta * lam * a
        src = eta ** 2 * (g[1:] @ Gam)
        decay = jnp.exp(-rate * dt)
        gain = jnp.where(rate * dt > 1e-12, (1 - decay) / jnp.where(rate > 0, rate, 1.0), dt)
        e = e * decay + src * gain
        return e, None

    def outer(e, _):
        P = 0.5 * e.sum()
        e, _ = lax.scan(step, e, None, length=every)
        return e, P

    e, Ps = lax.scan(outer, e0, None, length=steps // every)
    return Ps, e


def run_ode(e0, lam, Gam, P_grid, a_tab, g_tab, eta, T, steps=400_000, every=1):
    Ps, _ = _ode(jnp.asarray(e0), jnp.asarray(lam), jnp.asarray(Gam),
                 jnp.log(jnp.asarray(P_grid)), jnp.asarray(a_tab), jnp.asarray(g_tab),
                 float(eta), float(T), steps=steps, every=every)
    t = np.arange(steps // every) * (T / steps) * every
    keep = np.unique(np.concatenate([[0], np.round(np.geomspace(1, len(t) - 1, 600)).astype(int)]))
    return t[keep], np.asarray(Ps)[keep]


# --------------------------------------------------------------------------- #
# Adam on the dataset and on its Gaussian twin
# --------------------------------------------------------------------------- #
def record_steps(total, num=240):
    k = np.unique(np.round(np.geomspace(1, total, num)).astype(np.int64))
    return np.concatenate([[0], k])


@partial(jax.jit, static_argnames=("mode", "seeds", "total", "bias"))
def _adam(key, X, y, Lt, theta_star, resid, Kmat, epsK, mask, eta_p, beta1, beta2,
          *, mode, seeds, total, bias):
    D = theta_star.shape[0]
    th_s = theta_star.astype(jnp.float32)

    def sample(k):
        k1, k2 = jax.random.split(k)
        if mode == "real":
            idx = jax.random.randint(k1, (seeds,), 0, X.shape[0])
            return X[idx], y[idx]
        x = jax.random.normal(k1, (seeds, D), jnp.float32) @ Lt
        if mode == "twin_gauss":
            om = jnp.sqrt(jnp.mean(resid ** 2)) * jax.random.normal(k2, (seeds,), jnp.float32)
        else:
            om = resid[jax.random.randint(k2, (seeds,), 0, resid.shape[0])]
        return x, x @ th_s + om

    def excess(th):
        d = (th - th_s).astype(jnp.float64)
        return 0.5 * jnp.einsum("si,ij,sj->s", d, Kmat, d)

    def step(carry, inp):
        th, m, v, key = carry
        rec, k1 = inp
        key, k = jax.random.split(key)
        x, yy = sample(k)
        f = jnp.sum(x * th, axis=1) - yy
        g = x * f[:, None]
        m = beta1 * m + (1 - beta1) * g
        v = beta2 * v + (1 - beta2) * g * g
        if bias:   # standard (bias-corrected) Adam; identical in the D -> oo limit
            kk = (k1 + 1).astype(jnp.float32)
            th = th - eta_p * (m / (1 - beta1 ** kk)) / jnp.sqrt(v / (1 - beta2 ** kk) + epsK)
        else:
            th = th - eta_p * m / jnp.sqrt(v + epsK)
        P = lax.cond(rec, excess, lambda _: jnp.zeros(seeds), th)
        return (th, m, v, key), P

    z = jnp.zeros((seeds, D), jnp.float32)
    _, Ps = lax.scan(step, (z, z, z, key), (mask, jnp.arange(total)))
    return Ps


def run_adam(mode, X_dev, y_dev, Lt, theta, resid_dev, K, epsK, eta, beta1, beta2, T,
             seeds, key, bias=True):
    D = theta.shape[0]
    total = int(T * D)
    recs = record_steps(total)
    mask = np.zeros(total, bool)
    mask[recs[1:] - 1] = True
    Ps = _adam(key, X_dev, y_dev, Lt, theta, resid_dev, K, epsK, jnp.asarray(mask),
               jnp.float32(eta / D), jnp.float32(beta1), jnp.float32(beta2),
               mode=mode, seeds=seeds, total=total, bias=bias)
    Ps = np.asarray(Ps)[mask]
    P0 = float(0.5 * theta @ K @ theta)
    t = recs / D
    return t, np.concatenate([np.full((1, seeds), P0), Ps])


# --------------------------------------------------------------------------- #
# Commands
# --------------------------------------------------------------------------- #
def prepare(name):
    X, y, info = load(name)
    t0 = time.time()
    K, theta, resid = population(X, y)
    lam, W, d4, C = spectral(K)
    print(f"[{name}] population stats in {time.time() - t0:.1f}s  ({info})")
    return X, y, info, K, theta, resid, lam, W, d4, C


def cmd_diag(args):
    X, y, info, K, theta, resid, lam, W, d4, C = prepare(args.name)
    out = diagnostics(args.name, X, y, K, theta, resid, lam, W, d4, C)
    out["info"] = info
    RES.mkdir(parents=True, exist_ok=True)
    (RES / f"diag_{args.name}.json").write_text(json.dumps(out, indent=1))
    for k, v in out.items():
        print(f"  {k:24s} {v:.4g}" if isinstance(v, float) else f"  {k:24s} {v}")


def cmd_run(args):
    X, y, info, K, theta, resid, lam, W, d4, C = prepare(args.name)
    D = len(theta)
    sig2 = float(np.mean(resid ** 2))
    P0 = float(0.5 * theta @ K @ theta)
    if args.eta_rel:
        args.eta = float(args.eta_rel * np.sqrt(sig2 + 2 * P0) / np.mean(np.asarray(lam)))
        print(f"  eta_rel {args.eta_rel:g} -> eta {args.eta:.4g}")
    tag = (f"{args.name}_eta{args.eta:.3g}_b1{args.beta1:g}_b2{args.beta2:g}"
           f"_eps{args.eps:g}_T{args.T:g}{'_nobc' if args.no_bias_correction else ''}{args.tag}")
    print(f"[{tag}] D={D} P0={P0:.4g} sigma2={sig2:.4g}")

    # Coefficient tables on a P grid spanning the trajectory.
    P_grid = np.geomspace(max(P0, sig2) * 1e-4, P0 * 1.5, args.grid)
    t0 = time.time()
    tables = {}
    for noise in args.noise:
        a_d, a_i, g = chain_coefficients(P_grid, args.beta1, args.beta2, args.eps, resid,
                                         gauss=(noise == "gauss"), seed=args.seed)
        tables[noise] = (a_d, a_i, g)
        print(f"  coef[{noise}] {time.time() - t0:.1f}s  a_deriv/a_ibp at P0: "
              f"{a_d[-2]:.4g}/{a_i[-2]:.4g}  g(1) {g[-2, -1]:.4g}  g(.5) {g[-2, 8]:.4g}")
    Gam = gamma_table(lam, W, d4, C)
    e0 = np.asarray(lam * (W.T @ (d4 * theta)) ** 2)

    curves = {}
    for noise, (a_d, a_i, g) in tables.items():
        t_ode, P_ode = run_ode(e0, lam, Gam, P_grid, a_d, g, args.eta, args.T)
        curves[f"ode_{noise}"] = (t_ode, P_ode)

    epsK = jnp.asarray(args.eps * np.diag(np.asarray(K)), jnp.float32)
    X_dev = jnp.asarray(X)
    y_dev = jnp.asarray(y)
    r_dev = jnp.asarray(resid, jnp.float32)
    vals, vecs = jnp.linalg.eigh(K)
    Lt = (vecs * jnp.sqrt(jnp.clip(vals, 0))).T.astype(jnp.float32)   # x = z @ Lt
    key = jax.random.PRNGKey(args.seed + 1)
    for mode in args.modes:
        t1 = time.time()
        key, sub = jax.random.split(key)
        t, P = run_adam(mode, X_dev, y_dev, Lt, theta, r_dev, K, epsK, args.eta,
                        args.beta1, args.beta2, args.T, args.seeds, sub,
                        bias=not args.no_bias_correction)
        curves[mode] = (t, P)
        print(f"  sim[{mode}] {time.time() - t1:.1f}s  final P mean {P[-1].mean():.4g}")

    RES.mkdir(parents=True, exist_ok=True)
    np.savez(RES / f"run_{tag}.npz", P_grid=P_grid, sigma2=sig2, P0=P0, info=info,
             args=json.dumps(vars(args)),
             **{f"{k}_t": v[0] for k, v in curves.items()},
             **{f"{k}_P": v[1] for k, v in curves.items()},
             **{f"tab_{k}_a": v[0] for k, v in tables.items()},
             **{f"tab_{k}_aibp": v[1] for k, v in tables.items()},
             **{f"tab_{k}_g": v[2] for k, v in tables.items()})
    summarize(curves, tag)
    plot_run(curves, tag, info, sig2, args)


def rel_error(curves, sim, ode, tmin=0.0):
    t, P = curves[sim]
    to, Po = curves[ode]
    Pm = P.mean(axis=1)
    sel = t >= tmin
    Pi = np.exp(np.interp(t[sel], to, np.log(Po)))
    return float(np.max(np.abs(Pm[sel] - Pi) / Pi)), float(np.median(np.abs(Pm[sel] - Pi) / Pi))


def summarize(curves, tag):
    sims = [k for k in curves if not k.startswith("ode")]
    odes = [k for k in curves if k.startswith("ode")]
    for s in sims:
        for o in odes:
            mx, md = rel_error(curves, s, o, tmin=0.05)
            print(f"  {s:11s} vs {o:10s}: excess-risk rel. err  max {mx:.3f}  median {md:.3f}")


def plot_run(curves, tag, info, sig2, args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    FIGS.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.5, 5))
    colors = {"real": "C0", "twin_emp": "C2", "twin_gauss": "C4"}
    for k, (t, P) in curves.items():
        if k.startswith("ode"):
            continue
        lo, hi = np.percentile(P, [10, 90], axis=1)
        ax.fill_between(t[1:], lo[1:], hi[1:], color=colors[k], alpha=0.25, lw=0)
        ax.plot(t[1:], P.mean(1)[1:], color=colors[k], lw=1.8, label=f"Adam ({k})")
    styles = {"ode_emp": ("k", "--"), "ode_gauss": ("C3", ":")}
    for k, (t, P) in curves.items():
        if k.startswith("ode"):
            c, ls = styles[k]
            ax.plot(t[1:], P[1:], color=c, ls=ls, lw=2, label=k.replace("_", " "))
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("t = steps / D"); ax.set_ylabel(r"excess risk $R-\sigma^2/2$")
    ax.set_title(info, fontsize=9)
    ax.text(0.02, 0.03, rf"$\eta$={args.eta:g}, $\beta_1$={args.beta1:g}, "
            rf"$\beta_2$={args.beta2:g}, $\epsilon$={args.eps:g}, $\sigma^2/2$={sig2/2:.3g}",
            transform=ax.transAxes, fontsize=9)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGS / f"{tag}.png", dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("diag"); d.add_argument("name")
    r = sub.add_parser("run"); r.add_argument("name")
    r.add_argument("--eta", type=float, default=1.0)
    r.add_argument("--beta1", type=float, default=0.9)
    r.add_argument("--beta2", type=float, default=0.999)
    r.add_argument("--eps", type=float, default=0.0)
    r.add_argument("--eta-rel", type=float, default=0.0)
    r.add_argument("--T", type=float, default=20.0)
    r.add_argument("--seeds", type=int, default=16)
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--grid", type=int, default=24)
    r.add_argument("--noise", nargs="+", default=["emp", "gauss"])
    r.add_argument("--modes", nargs="+", default=["real", "twin_emp"])
    r.add_argument("--no-bias-correction", action="store_true")
    r.add_argument("--tag", default="")
    args = ap.parse_args()
    {"diag": cmd_diag, "run": cmd_run}[args.cmd](args)


if __name__ == "__main__":
    main()
