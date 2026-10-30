#!/usr/bin/env python3
"""Adam under a Haar-rotated covariance.

Adam is coordinate-wise, so it sees the covariance ``K`` only through
``D = sqrt(diag K)``: the drift operator is ``Kbar = D^{-1} K`` and the diffusion
uses the coordinate correlations ``C = D^{-1} K D^{-1}``. For a diagonal
``K = Lambda`` this gives ``Kbar = Lambda^{1/2}`` (a square-root preconditioner).
For ``K = O Lambda O^T`` with Haar ``O``, ``diag K ~ mean(lambda)`` and
``Kbar ~ K / sqrt(mean lambda)``: the preconditioning is lost. The diffusion also
changes: off-diagonal noise correlations are ``tau C_ik`` with ``tau < sigma-hat``
(``2/pi`` of it for signSGD, the arcsine law), so a fraction
``1 - tau/sigma-hat`` of Adam's noise is isotropic in parameter space.

Subcommands (results in ``--output-dir``):

* ``risk``  : noisy linear regression, same spectrum, diagonal vs Haar ``K``;
  Adam / SGD runs against their ODEs, per-mode errors and ``D``-scaling.
* ``noise`` : frozen-parameter long-run covariance of the Adam direction
  against ``C_ik``; ``tau / sigma-hat`` as a function of ``beta2``; stationary
  per-mode errors against the ODE and two ablated diffusion models.
* ``plot``  : figures ``haar_rotation.{pdf,png}`` and ``haar_noise.{pdf,png}``.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", "/tmp/adam-matplotlib")

import config
config.enable_x64()

import jax
import jax.numpy as jnp
import matplotlib
import numpy as np
from jax import lax

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import discounts
import dynamics
import problems
import simulate
from experiments.plot_style import apply_paper_style, polish_axis
from utils import haar_orthogonal

apply_paper_style()

GEOMETRIES = ("diagonal", "haar")
COLORS = {"diagonal": "#1f77b4", "haar": "#d62728", "sgd": "#7f7f7f"}


def log_uniform_spectrum(d, lam_min, lam_max):
    lam = jnp.exp(jnp.linspace(np.log(lam_min), np.log(lam_max), d))
    return lam / lam.mean()


def instance(d, args, seed=0):
    """Same spectrum and same whitened student/teacher in both geometries.

    The Haar instance is the exact rotation ``x -> O x``, ``Theta -> O Theta`` of
    the diagonal one, so ``B`` and every rotation-invariant method (SGD) agree.
    """
    lam = log_uniform_spectrum(d, args.lam_min, args.lam_max)
    k_o, k_s, k_t = jax.random.split(jax.random.PRNGKey(seed + d), 3)
    star = jax.random.normal(k_s, (d, 1))
    star = star / jnp.sqrt(jnp.sum(lam[:, None] * star ** 2))
    theta = args.init_scale * jax.random.normal(k_t, (d, 1)) / jnp.sqrt(d)
    O = haar_orthogonal(k_o, d)
    out = {}
    for geometry in GEOMETRIES:
        if geometry == "diagonal":
            factor, t0, ts = jnp.sqrt(lam), theta, star
        else:
            factor, t0, ts = O * jnp.sqrt(lam)[None, :], O @ theta, O @ star
        out[geometry] = (simulate.MixtureData(factor, jnp.zeros((d, 1)), ts), t0)
    return lam, out


def mode_basis(cov, rho):
    return jnp.diag(cov ** 0.25) if rho is None else rho


def per_mode_error(proj, modes):
    """``nu_j (rho_j^T (Theta - Theta*))^2`` from recorded ``rho^T Theta``."""
    return modes.nu * (proj[..., 0] - modes.teacher[:, 0]) ** 2


def binned(values, keys, edges):
    idx = np.clip(np.digitize(keys, edges) - 1, 0, len(edges) - 2)
    count = np.bincount(idx, minlength=len(edges) - 1)
    total = np.stack([np.bincount(idx, v, minlength=len(edges) - 1)
                      for v in values.reshape(-1, values.shape[-1])])
    return total.reshape(values.shape[:-1] + (-1,)) / np.maximum(count, 1)


def solve(problem, data, theta0, args, *, beta1, beta2, eps, T, sgd=False,
          variant=None, draws=None, layout=None, stein=False):
    d = theta0.shape[0]
    cov = simulate.mixture_covariance(data)
    build = dynamics.sgd_modes if sgd else dynamics.adam_modes
    modes, rho = build(cov, theta0, data.teacher, data.means)
    if variant == "aligned":        # tau = sigma: noise shaped like the data
        modes = modes._replace(w_iso=modes.w_cov)
    elif variant == "isotropic":    # tau = 0: noise ignores coordinate correlations
        modes = modes._replace(w_cov=modes.w_iso)
    ode = dynamics.run_mixture_ode(
        problem, modes, args.eta, T, beta1=beta1, beta2=beta2, eps=eps,
        draws=draws, layout=layout, dimension=d, dt=args.dt,
        record_every=args.ode_record, sgd=sgd, stein=stein)
    return ode, modes, rho


# --------------------------------------------------------------------------- #
# risk: diagonal vs Haar
# --------------------------------------------------------------------------- #
def run_risk(args):
    problem = problems.make_mixture_linreg(1, args.noise_std)
    layout = discounts.path_layout(args.beta1, args.beta2, args.window)
    draws = discounts.draw_paths(jax.random.PRNGKey(1), args.paths, layout,
                                 batch=1, inputs=2, outputs=1)
    record = args.record_per_unit
    results = {}
    for d in args.dimensions:
        lam, cases = instance(d, args)
        steps = int(round(args.horizon * d))
        keys = jax.random.split(jax.random.PRNGKey(100 + d), args.runs)
        main = d == max(args.dimensions)
        for geometry, (data, theta0) in cases.items():
            t0 = time.time()
            ode, modes, rho = solve(problem, data, theta0, args, beta1=args.beta1,
                                    beta2=args.beta2, eps=args.epsilon,
                                    T=args.horizon, draws=draws, layout=layout)
            cov = simulate.mixture_covariance(data)
            sim = simulate.build_mixture_adam(problem, data, args.eta / d,
                                              beta1=args.beta1, beta2=args.beta2,
                                              eps=args.epsilon)
            basis = mode_basis(cov, rho) if main else None
            B, O, proj = simulate.run_mixture_many(sim, data, theta0, steps,
                                                   record_every=d // record,
                                                   keys=keys, basis=basis)
            risk = jax.vmap(jax.vmap(problem.risk))(B, O)
            tag = f"{geometry}_D{d}"
            results[f"adam_{tag}"] = np.asarray(risk)
            results[f"ode_{tag}"] = np.asarray(ode["risk"])
            results[f"time_{tag}"] = np.arange(risk.shape[1]) / record
            results[f"odetime_{tag}"] = np.asarray(ode["time"])
            if main:
                results[f"nu_{geometry}"] = np.asarray(modes.nu)
                results[f"lam_{geometry}"] = np.asarray(lam)
                results[f"adam_modes_{geometry}"] = np.asarray(
                    per_mode_error(proj, modes))                       # (runs, rec, J)
                results[f"ode_modes_{geometry}"] = np.asarray(ode["mode_error"])
                sode, smodes, _ = solve(problem, data, theta0, args, beta1=0.0,
                                        beta2=0.0, eps=0.0, T=args.horizon,
                                        sgd=True, draws=draws, layout=layout)
                ssim = simulate.build_mixture_sgd(problem, data, args.eta / d)
                sB, sO, _ = simulate.run_mixture_many(ssim, data, theta0, steps,
                                                      record_every=d // record, keys=keys)
                results[f"sgd_{geometry}"] = np.asarray(jax.vmap(jax.vmap(problem.risk))(sB, sO))
                results[f"sgdode_{geometry}"] = np.asarray(sode["risk"])
                results[f"nusgd_{geometry}"] = np.asarray(smodes.nu)
            gap = np.abs(results[f"adam_{tag}"].mean(0) - matched(results, tag)).max()
            print(f"D={d:5d} {geometry:8s}: mean gap {gap:.4f}  [{time.time()-t0:.0f}s]",
                  flush=True)
    np.savez_compressed(args.output_dir / "haar_risk.npz", **results)


def matched(results, tag):
    """ODE risk at the optimizer's record times."""
    return np.interp(results[f"time_{tag}"], results[f"odetime_{tag}"], results[f"ode_{tag}"])


# --------------------------------------------------------------------------- #
# noise: diffusion structure
# --------------------------------------------------------------------------- #
NOISE_BETAS = ((0.0, 0.0, 1e-6), (0.5, 0.5, 1e-6), (0.9, 0.95, 1e-6))
SWEEP_BETA1 = (0.0, 0.5, 0.9)


def frozen_covariance(data, theta, problem, beta1, beta2, eps, *, block, burn,
                      chains, key):
    """Long-run covariance ``Cov(sum_t u_t) / block`` of the Adam direction
    ``u_t = m_t / sqrt(v_t + eps K_ii)`` at a frozen parameter."""
    d = theta.shape[0]
    factor = data.factor
    stability = eps * jnp.sum(factor ** 2, axis=1)
    W = jnp.concatenate([theta, data.teacher], axis=1)

    def chain(key):
        def step(carry, k):
            m, v, acc = carry
            k1, k2 = jax.random.split(k)
            x = factor @ jax.random.normal(k1, (d,))
            g = x * problem.f(x @ W, 0, jax.random.normal(k2, (1,)))[0]
            m = beta1 * m + (1 - beta1) * g
            v = beta2 * v + (1 - beta2) * g ** 2
            return (m, v, acc + m / jnp.sqrt(v + stability)), None

        zero = jnp.zeros(d)
        (m, v, _), _ = lax.scan(step, (zero, zero, zero), jax.random.split(key, burn))
        (_, _, acc), _ = lax.scan(step, (m, v, zero),
                                  jax.random.split(jax.random.fold_in(key, 1), block))
        return acc

    sums = jax.jit(jax.vmap(chain))(jax.random.split(key, chains))
    return np.cov(np.asarray(sums).T) / block


def run_noise(args):
    problem = problems.make_mixture_linreg(1, args.noise_std)
    results = {}

    # (a) frozen-parameter covariance vs coordinate correlation
    d = args.frozen_dimension
    lam, cases = instance(d, args, seed=7)
    data, theta = cases["haar"]
    K = simulate.mixture_covariance(data)
    scale = jnp.sqrt(jnp.diag(K))
    C = np.asarray(K / scale[:, None] / scale[None, :])
    W = jnp.concatenate([theta, data.teacher], axis=1)
    B = W.T @ K @ W
    off = ~np.eye(d, dtype=bool)
    edges = np.quantile(C[off], np.linspace(0, 1, args.scatter_bins + 1))
    results["C_bins"] = binned(C[off][None], C[off], edges)[0]
    for i, (b1, b2, eps) in enumerate(NOISE_BETAS):
        t0 = time.time()
        S = frozen_covariance(data, theta, problem, b1, b2, eps, block=args.block,
                              burn=args.burn, chains=args.chains,
                              key=jax.random.PRNGKey(20 + i))
        layout = discounts.path_layout(b1, b2, 4096)
        draws = discounts.draw_paths(jax.random.PRNGKey(3), 256, layout, batch=1,
                                     inputs=2, outputs=1)
        co = discounts.stationary_coefficients(B, jnp.zeros((2, 1)), problem.priors,
                                               draws, problem.f, b1, b2, eps,
                                               burn=layout[0], tail=layout[2])
        slope = np.sum(S[off] * C[off]) / np.sum(C[off] ** 2)
        results[f"S_bins_{i}"] = binned(S[off][None], C[off], edges)[0]
        results[f"S_diag_{i}"] = np.diag(S)
        results[f"slope_{i}"] = slope
        results[f"sigma_{i}"] = float(co["sigma"][0, 0])
        results[f"tau_{i}"] = float(co["tau"][0, 0])
        print(f"beta=({b1},{b2}): diag {np.diag(S).mean():.4f} sigma {results[f'sigma_{i}']:.4f}"
              f" | slope {slope:.4f} tau {results[f'tau_{i}']:.4f}  [{time.time()-t0:.0f}s]",
              flush=True)

    # (b) tau / sigma as a function of beta2
    beta2s = np.array(args.beta2_grid)
    for b1 in SWEEP_BETA1:
        ratio = []
        for b2 in beta2s:
            if b1 > 0 and b2 == 0:      # g_{t-l} / |g_t| terms: sigma-hat is infinite
                ratio.append(np.nan)
                continue
            layout = discounts.path_layout(b1, b2, 2048)
            draws = discounts.draw_paths(jax.random.PRNGKey(4), 128, layout, batch=1,
                                         inputs=2, outputs=1)
            co = discounts.stationary_coefficients(B, jnp.zeros((2, 1)), problem.priors,
                                                   draws, problem.f, b1, b2, 1e-6,
                                                   burn=layout[0], tail=layout[2])
            ratio.append(float(co["tau"][0, 0] / co["sigma"][0, 0]))
        results[f"ratio_beta1_{b1}"] = np.array(ratio)
    results["beta2_grid"] = beta2s

    # (c) stationary per-mode errors (Haar), ODE vs ablated diffusions
    d = args.mode_dimension
    lam, cases = instance(d, args, seed=11)
    T = args.mode_horizon
    steps = int(T * d)
    record = args.record_per_unit
    for i, (b1, b2, eps) in enumerate((NOISE_BETAS[0], NOISE_BETAS[2])):
        layout = discounts.path_layout(b1, b2, args.window)
        draws = discounts.draw_paths(jax.random.PRNGKey(5), args.paths, layout,
                                     batch=1, inputs=2, outputs=1)
        for geometry, (data, theta0) in cases.items():
            t0 = time.time()
            variants = (None, "aligned", "isotropic") if geometry == "haar" else (None,)
            for variant in variants:
                ode, modes, rho = solve(problem, data, theta0, args, beta1=b1,
                                        beta2=b2, eps=eps, T=T, variant=variant,
                                        draws=draws, layout=layout, stein=True)
                results[f"stat_ode_{i}_{geometry}_{variant}"] = np.asarray(ode["mode_error"][-1])
                results[f"stat_oderisk_{i}_{geometry}_{variant}"] = np.asarray(ode["risk"])
            cov = simulate.mixture_covariance(data)
            sim = simulate.build_mixture_adam(problem, data, args.eta / d, beta1=b1,
                                              beta2=b2, eps=eps)
            Bs, Os, proj = simulate.run_mixture_many(
                sim, data, theta0, steps, record_every=d // record,
                keys=jax.random.split(jax.random.PRNGKey(50 + i), args.mode_runs),
                basis=mode_basis(cov, rho))
            tail = int(args.average_window * record)
            errs = per_mode_error(proj[:, -tail:], modes)          # (runs, tail, J)
            results[f"stat_adam_{i}_{geometry}"] = np.asarray(errs.mean((0, 1)))
            results[f"stat_adamrisk_{i}_{geometry}"] = np.asarray(
                jax.vmap(jax.vmap(problem.risk))(Bs, Os))
            results[f"stat_nu_{i}_{geometry}"] = np.asarray(modes.nu)
            results[f"stat_lam_{i}_{geometry}"] = np.asarray(lam)
            print(f"stationary beta=({b1},{b2}) {geometry}: [{time.time()-t0:.0f}s]", flush=True)
    results["stat_odetime"] = np.asarray(ode["time"])
    np.savez_compressed(args.output_dir / "haar_noise.npz", **results)


# --------------------------------------------------------------------------- #
# plots
# --------------------------------------------------------------------------- #
def band(ax, t, runs, color, label=None):
    lo, hi = np.percentile(runs, [10, 90], axis=0)
    ax.fill_between(t, lo, hi, color=color, alpha=0.3, linewidth=0, label=label)


def plot(args):
    r = dict(np.load(args.output_dir / "haar_risk.npz"))
    dmax = max(args.dimensions)
    fig, axes = plt.subplots(1, 4, figsize=(24, 5.6), constrained_layout=True)

    ax = axes[0]
    for g in GEOMETRIES:
        tag = f"{g}_D{dmax}"
        band(ax, r[f"time_{tag}"], r[f"adam_{tag}"], COLORS[g], f"Adam, {g} $K$")
        ax.plot(r[f"odetime_{tag}"], r[f"ode_{tag}"], color=COLORS[g], lw=2.2)
    band(ax, r[f"time_diagonal_D{dmax}"], r["sgd_diagonal"], COLORS["sgd"], "SGD (either $K$)")
    ax.plot(r[f"odetime_diagonal_D{dmax}"], r["sgdode_diagonal"], color="k", lw=1.6, ls=":")
    ax.plot([], [], color="k", lw=2.2, label="ODE")
    ax.axhline(args.noise_std ** 2 / 2, color="0.5", lw=1, ls="--")
    ax.set_yscale("log")
    ax.set_xlabel(r"$t=k/D$")
    ax.set_ylabel("risk")
    ax.set_title(f"Same spectrum, $D={dmax}$")
    ax.legend(frameon=False, fontsize=12)

    ax = axes[1]
    rank = np.arange(1, dmax + 1) / dmax
    lam = np.sort(r["lam_haar"])[::-1]
    ax.plot(rank, np.sort(r["nu_diagonal"])[::-1], color=COLORS["diagonal"],
            label=r"Adam, diagonal: $\sqrt{\lambda}$")
    ax.plot(rank, np.sort(r["nu_haar"])[::-1], color=COLORS["haar"],
            label=r"Adam, Haar: eig$(\overline{K})$")
    ax.plot(rank, lam / np.sqrt(lam.mean()), color="k", ls="--", lw=1.4,
            label=r"$\lambda/\sqrt{\bar\lambda}$")
    ax.plot(rank, lam, color=COLORS["sgd"], ls=":", label=r"SGD: $\lambda$")
    ax.set_yscale("log")
    ax.set_xlabel("mode rank / $D$")
    ax.set_ylabel("drift rate of mode")
    ax.set_title(r"Spectrum of $\overline{K}=\mathrm{diag}(K)^{-1/2}K$")
    ax.legend(frameon=False, fontsize=12)

    ax = axes[2]
    record = args.record_per_unit
    edges = np.geomspace(lam.min(), lam.max() * 1.0001, 25)
    centers = np.sqrt(edges[1:] * edges[:-1])
    for g in GEOMETRIES:
        # modes indexed by the data eigenvalue they carry: for Haar, nu_j ~ lam_j / sqrt(mean lam)
        key_lam = np.sort(r["lam_haar"])
        order = np.argsort(r[f"nu_{g}"])
        for t_show, marker in ((1.0, "o"), (args.horizon, "s")):
            k = int(round(t_show * record))
            adam = r[f"adam_modes_{g}"][:, k, order].mean(0)
            ko = int(round(t_show / (args.dt * args.ode_record)))
            ode = r[f"ode_modes_{g}"][ko, order]
            ax.plot(centers, dmax * binned(ode[None], key_lam, edges)[0], color=COLORS[g], lw=2)
            ax.plot(centers, dmax * binned(adam[None], key_lam, edges)[0], marker, color=COLORS[g],
                    ms=6, mfc="none" if marker == "s" else COLORS[g], ls="none")
    ax.plot([], [], "ko", label="Adam, $t=1$")
    ax.plot([], [], "ks", mfc="none", label=f"Adam, $t={args.horizon:g}$")
    ax.plot([], [], "k-", label="ODE")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"data eigenvalue $\lambda$ of mode")
    ax.set_ylabel(r"$D\times$ mode excess risk")
    ax.set_title("Per-mode error")
    ax.legend(frameon=False, fontsize=12)

    ax = axes[3]
    dims = np.array(sorted(args.dimensions))
    for g in GEOMETRIES:
        per_run = []
        for d in dims:
            tag = f"{g}_D{d}"
            per_run.append(np.max(np.abs(r[f"adam_{tag}"] - matched(r, tag)[None]), axis=1))
        per_run = np.array(per_run)
        med = np.median(per_run, axis=1)
        q1, q3 = np.percentile(per_run, [25, 75], axis=1)
        ax.errorbar(dims, med, yerr=[med - q1, q3 - med], marker="o", color=COLORS[g],
                    capsize=4, label=g)
    ax.plot(dims, med[0] * np.sqrt(dims[0] / dims), "k--", lw=1.4, label=r"$\propto D^{-1/2}$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("$D$")
    ax.set_ylabel(r"$\sup_t|\mathcal{R}(\theta_{tD})-\mathscr{R}(t)|$")
    ax.set_title("Concentration (median, IQR)")
    ax.legend(frameon=False, fontsize=12)
    for ax in axes:
        polish_axis(ax)
    save(fig, args.output_dir / "haar_rotation")

    n = dict(np.load(args.output_dir / "haar_noise.npz"))
    fig, axes = plt.subplots(1, 4, figsize=(24, 5.6), constrained_layout=True)
    ax = axes[0]
    cols = plt.cm.viridis(np.linspace(0.1, 0.8, len(NOISE_BETAS)))
    for i, ((b1, b2, _), c) in enumerate(zip(NOISE_BETAS, cols)):
        sig, tau = n[f"sigma_{i}"], n[f"tau_{i}"]
        x = n["C_bins"]
        ax.plot(x, n[f"S_bins_{i}"] / sig, "o", color=c, ms=5,
                label=fr"$\beta=({b1:g},{b2:g})$: $\tau/\hat\sigma={tau/sig:.2f}$")
        ax.plot(x, tau / sig * x, color=c, lw=2)
    ax.plot(x, x, "k--", lw=1.4, label=r"slope 1 ($\tau=\hat\sigma$, SGD-like)")
    ax.set_xlabel(r"data correlation $C_{ik}$")
    ax.set_ylabel(r"noise correlation $\Sigma_{ik}/\hat\sigma$")
    ax.set_title("Frozen-parameter noise, Haar $K$")
    ax.legend(frameon=False, fontsize=11)

    ax = axes[1]
    for b1, c in zip(SWEEP_BETA1, cols):
        ax.plot(n["beta2_grid"], n[f"ratio_beta1_{b1}"], "o-", color=c, label=fr"$\beta_1={b1:g}$")
    for i, ((b1, b2, _), c) in enumerate(zip(NOISE_BETAS, cols)):
        ax.plot(b2, n[f"slope_{i}"] / np.mean(n[f"S_diag_{i}"]), "*", color=c, mec="k",
                ms=16, label="measured slope / diagonal (panel 1)" if i == 0 else None)
    ax.axhline(2 / np.pi, color="0.4", ls=":", lw=1.4)
    ax.text(0.02, 2 / np.pi + 0.01, r"$2/\pi$ (signSGD)", fontsize=12)
    ax.set_xlabel(r"$\beta_2$")
    ax.set_ylabel(r"$\tau/\hat\sigma$")
    ax.set_title("Data-aligned fraction of the noise")
    ax.legend(frameon=False, fontsize=12)

    for i, ax in zip((0, 1), axes[2:]):
        b1, b2, _ = NOISE_BETAS[0 if i == 0 else 2]
        lam = n[f"stat_lam_{i}_haar"]
        edges = np.geomspace(lam.min(), lam.max() * 1.0001, 25)
        centers = np.sqrt(edges[1:] * edges[:-1])
        key_lam = np.sort(lam)
        d = len(lam)
        for g in GEOMETRIES:
            order = np.argsort(n[f"stat_nu_{i}_{g}"])
            ax.plot(centers, d * binned(n[f"stat_adam_{i}_{g}"][order][None], key_lam, edges)[0],
                    "o", color=COLORS[g], ms=6, label=f"Adam, {g}")
            ax.plot(centers, d * binned(n[f"stat_ode_{i}_{g}_None"][order][None], key_lam, edges)[0],
                    color=COLORS[g], lw=2.2, label=f"ODE, {g}")
        order = np.argsort(n[f"stat_nu_{i}_haar"])
        for variant, ls, label in (("aligned", "--", r"Haar ODE, $\tau=\hat\sigma$"),
                                   ("isotropic", ":", r"Haar ODE, $\tau=0$")):
            ax.plot(centers, d * binned(n[f"stat_ode_{i}_haar_{variant}"][order][None], key_lam, edges)[0],
                    color="k", ls=ls, lw=1.6, label=label)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"data eigenvalue $\lambda$ of mode")
        ax.set_ylabel(r"$D\times$ stationary mode excess risk")
        ax.set_title(fr"Stationary modes, $\beta=({b1:g},{b2:g})$")
        ax.legend(frameon=False, fontsize=11)
    for ax in axes:
        polish_axis(ax)
    save(fig, args.output_dir / "haar_noise")


def save(fig, path):
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved -> {path}.{{pdf,png}}")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=["risk", "noise", "plot", "all"])
    p.add_argument("--dimensions", type=int, nargs="+", default=[500, 1000, 2000, 4000])
    p.add_argument("--runs", type=int, default=16)
    p.add_argument("--horizon", type=float, default=12.0)
    p.add_argument("--eta", type=float, default=0.5)
    p.add_argument("--beta1", type=float, default=0.9)
    p.add_argument("--beta2", type=float, default=0.95)
    p.add_argument("--epsilon", type=float, default=1e-2)
    p.add_argument("--noise-std", type=float, default=0.5)
    p.add_argument("--lam-min", type=float, default=0.05)
    p.add_argument("--lam-max", type=float, default=5.0)
    p.add_argument("--init-scale", type=float, default=0.2)
    p.add_argument("--dt", type=float, default=0.025)
    p.add_argument("--ode-record", type=int, default=5)
    p.add_argument("--record-per-unit", type=int, default=8)
    p.add_argument("--paths", type=int, default=256)
    p.add_argument("--window", type=int, default=1024)
    p.add_argument("--frozen-dimension", type=int, default=500)
    p.add_argument("--block", type=int, default=3000)
    p.add_argument("--burn", type=int, default=400)
    p.add_argument("--chains", type=int, default=8192)
    p.add_argument("--scatter-bins", type=int, default=24)
    p.add_argument("--beta2-grid", type=float, nargs="+",
                   default=[0.0, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99])
    p.add_argument("--mode-dimension", type=int, default=2000)
    p.add_argument("--mode-horizon", type=float, default=60.0)
    p.add_argument("--mode-runs", type=int, default=16)
    p.add_argument("--average-window", type=float, default=20.0)
    p.add_argument("--output-dir", type=Path, default=ROOT / "figs" / "experiments")
    return p.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.command in ("risk", "all"):
        run_risk(args)
    if args.command in ("noise", "all"):
        run_noise(args)
    if args.command in ("plot", "all"):
        plot(args)


if __name__ == "__main__":
    main()
