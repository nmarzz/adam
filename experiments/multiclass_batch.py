#!/usr/bin/env python3
"""Multi-class Adam with mini-batches (gmm_adam.tex, sections gmm_batch / examples).

A 5-class Gaussian mixture with unequal priors and class-mean norms, softmax
regression with ``M = C`` logits, batch sizes ``b`` fixed as ``D`` grows. The ODE
uses the batch coefficients ``(phi_b, chi_b, sigma_b)`` and, for a dense
(Haar-rotated) covariance, the off-diagonal diffusion coefficient ``tau_b``.

Subcommands (results in ``--output-dir``):

* ``batch``    : Adam and SGD for ``b in {1, 4, 16, 64}``, diagonal ``K``.
* ``haar``     : the same mixture with a Haar-rotated ``K``, and a softmax
  teacher--student problem (matrix-valued ``tau``) on Haar ``K``.
* ``scaling``  : mean-risk gap to the ODE against ``D`` for several ``b``
  (the ``O(b/D)`` finite-size correction).
* ``drift``    : coefficients against ``b`` and a frozen-parameter check of the
  batch drift ``Kbar Theta phi_b + D^{-1} Xbar chi_b``.
* ``plot``     : figures ``multiclass_batch``, ``multiclass_scaling``, ``multiclass_haar``.
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

PRIORS = (0.3, 0.25, 0.2, 0.15, 0.1)
MEAN_NORMS = (2.0, 1.8, 1.6, 1.4, 1.2)


def spectrum(d):
    """Anisotropic profile of ``experiments/diagonal_linreg.py``, ``K_jj in [0.25, 2]``."""
    s = jnp.linspace(0.0, 1.0, d)
    return 0.25 + 1.75 * (jnp.exp(-4 * s) - jnp.exp(-4)) / (1 - jnp.exp(-4))


def mixture(d, geometry="diagonal", seed=0):
    """Means ``r_c v_c`` on a regular simplex in a random 5-dim subspace."""
    C = len(PRIORS)
    k_q, k_t, k_o = jax.random.split(jax.random.PRNGKey(seed + d), 3)
    Q, _ = jnp.linalg.qr(jax.random.normal(k_q, (d, C)))
    V = jnp.eye(C) - 1.0 / C
    V = V / jnp.linalg.norm(V, axis=1, keepdims=True)
    means = Q @ (V.T * jnp.asarray(MEAN_NORMS)[None, :])
    theta0 = 0.4 * jax.random.normal(k_t, (d, C)) / jnp.sqrt(d)
    lam = spectrum(d)
    factor = jnp.sqrt(lam) if geometry == "diagonal" else \
        haar_orthogonal(k_o, d) * jnp.sqrt(lam)[None, :]
    return simulate.MixtureData(factor, means, jnp.zeros((d, 0))), theta0


def coefficient_draws(args, batch, inputs, outputs, seed=0):
    layout = discounts.path_layout(args.beta1, args.beta2, args.window)
    paths = args.paths if batch <= 16 else args.paths // 2
    return layout, discounts.draw_paths(jax.random.PRNGKey(seed), paths, layout,
                                        batch=batch, inputs=inputs, outputs=outputs)


def ode_and_runs(problem, data, theta0, args, *, batch, runs, sgd=False, seed=0,
                 eta=None, horizon=None, record_per_unit=None):
    d = theta0.shape[0]
    eta = args.eta if eta is None else eta
    horizon = args.horizon if horizon is None else horizon
    record = args.record_per_unit if record_per_unit is None else record_per_unit
    cov = simulate.mixture_covariance(data)
    modes, _ = (dynamics.sgd_modes if sgd else dynamics.adam_modes)(
        cov, theta0, data.teacher, data.means)
    layout, draws = coefficient_draws(args, batch, theta0.shape[1] + data.teacher.shape[1],
                                      problem.outputs, seed)
    ode = dynamics.run_mixture_ode(problem, modes, eta, horizon, beta1=args.beta1,
                                   beta2=args.beta2, eps=args.epsilon, draws=draws,
                                   layout=layout, dimension=d, dt=args.dt,
                                   record_every=int(round(1 / (record * args.dt))), sgd=sgd)
    if sgd:
        sim = simulate.build_mixture_sgd(problem, data, eta / d, batch=batch)
    else:
        sim = simulate.build_mixture_adam(problem, data, eta / d, beta1=args.beta1,
                                          beta2=args.beta2, eps=args.epsilon, batch=batch)
    keys = jax.random.split(jax.random.PRNGKey(1000 + 7 * batch + d + seed), runs)
    B, O, _ = simulate.run_mixture_many(sim, data, theta0, int(round(horizon * d)),
                                        record_every=d // record, keys=keys)
    out = {"time": np.arange(B.shape[1]) / record,
           "risk": np.asarray(jax.vmap(jax.vmap(problem.risk))(B, O)),
           "ode_time": np.asarray(ode["time"]), "ode_risk": np.asarray(ode["risk"])}
    if problem.error is not None:
        out["error"] = np.asarray(jax.vmap(jax.vmap(problem.error))(B, O))
        out["ode_error"] = np.asarray(ode["error"])
    return out


def gap(out):
    ode = np.interp(out["time"], out["ode_time"], out["ode_risk"])
    mean = out["risk"].mean(0)
    se = out["risk"].std(0) / np.sqrt(len(out["risk"]))
    i = np.argmax(np.abs(mean - ode))
    return float(np.abs(mean - ode)[i]), float(se[i])


def store(results, prefix, out):
    for k, v in out.items():
        results[f"{prefix}_{k}"] = v


# --------------------------------------------------------------------------- #
def run_batch(args):
    problem = problems.make_gmm_softmax(PRIORS)
    d = args.dimension
    data, theta0 = mixture(d)
    results = {}
    for b in args.batches:
        for method, sgd, eta in (("adam", False, args.eta), ("sgd", True, args.sgd_eta)):
            t0 = time.time()
            out = ode_and_runs(problem, data, theta0, args, batch=b, runs=args.runs,
                               sgd=sgd, eta=eta)
            store(results, f"{method}_b{b}", out)
            print(f"{method} b={b:3d}: gap {gap(out)[0]:.4f} (se {gap(out)[1]:.4f})"
                  f"  [{time.time()-t0:.0f}s]", flush=True)
    np.savez_compressed(args.output_dir / "multiclass_batch.npz", **results)


def run_haar(args):
    results = {}
    problem = problems.make_gmm_softmax(PRIORS)
    d = args.haar_dimension
    for geometry in ("diagonal", "haar"):
        data, theta0 = mixture(d, geometry, seed=1)
        for b in args.haar_batches:
            t0 = time.time()
            out = ode_and_runs(problem, data, theta0, args, batch=b, runs=args.runs)
            store(results, f"gmm_{geometry}_b{b}", out)
            print(f"gmm {geometry} b={b}: gap {gap(out)[0]:.4f} (se {gap(out)[1]:.4f})"
                  f"  [{time.time()-t0:.0f}s]", flush=True)

    # softmax teacher--student, C=4 classes (M = M* = 3 logits), Haar K
    classes = args.teacher_classes
    rank = classes - 1
    problem = problems.make_mixture_softmax_teacher(classes)
    lam = spectrum(d)
    k_o, k_s, k_t = jax.random.split(jax.random.PRNGKey(5), 3)
    O = haar_orthogonal(k_o, d)
    white = jnp.linalg.qr(jax.random.normal(k_s, (d, 2 * rank)))[0]
    teacher = O @ (2.0 * white[:, :rank] / jnp.sqrt(lam)[:, None])
    theta0 = O @ (0.5 * white[:, rank:] / jnp.sqrt(lam)[:, None])
    data = simulate.MixtureData(O * jnp.sqrt(lam)[None, :], jnp.zeros((d, 1)), teacher)
    for b in args.teacher_batches:
        t0 = time.time()
        out = ode_and_runs(problem, data, theta0, args, batch=b, runs=args.runs,
                           eta=args.teacher_eta)
        store(results, f"teacher_b{b}", out)
        print(f"teacher b={b}: gap {gap(out)[0]:.4f} (se {gap(out)[1]:.4f})"
              f"  [{time.time()-t0:.0f}s]", flush=True)
    np.savez_compressed(args.output_dir / "multiclass_haar.npz", **results)


def run_scaling(args):
    problem = problems.make_gmm_softmax(PRIORS)
    results = {}
    for b in args.scaling_batches:
        for d in args.scaling_dimensions:
            t0 = time.time()
            data, theta0 = mixture(d, seed=2)
            runs = args.runs if d <= 4000 else args.runs // 2
            out = ode_and_runs(problem, data, theta0, args, batch=b, runs=runs)
            g, se = gap(out)
            results[f"gap_b{b}_D{d}"] = np.array([g, se])
            print(f"scaling b={b:3d} D={d:5d}: gap {g:.4f} (se {se:.4f})"
                  f"  [{time.time()-t0:.0f}s]", flush=True)
    np.savez_compressed(args.output_dir / "multiclass_scaling.npz", **results)


def frozen_drift(problem, data, theta, args, batch, *, steps, chains, burn=500):
    """Mean Adam direction ``E[m / sqrt(v + eps K_ii)]`` at a frozen parameter."""
    d, M = theta.shape
    stability = args.epsilon * simulate.mixture_covariance(data)[:, None]

    def chain(key):
        def step(carry, k):
            m, v, acc = carry
            g = simulate._mixture_grad(problem, data, theta, k, batch)
            m = args.beta1 * m + (1 - args.beta1) * g
            v = args.beta2 * v + (1 - args.beta2) * g ** 2
            return (m, v, acc + m / jnp.sqrt(v + stability)), None

        zero = jnp.zeros((d, M))
        (m, v, _), _ = lax.scan(step, (zero, zero, zero), jax.random.split(key, burn))
        (_, _, acc), _ = lax.scan(step, (m, v, zero),
                                  jax.random.split(jax.random.fold_in(key, 1), steps))
        return acc / steps

    return jax.jit(jax.vmap(chain))(jax.random.split(jax.random.PRNGKey(batch), chains)).mean(0)


def run_drift(args):
    problem = problems.make_gmm_softmax(PRIORS)
    d = args.drift_dimension
    data, theta0 = mixture(d, seed=3)
    lam = simulate.mixture_covariance(data)
    # a point mid-way through training: logits aligned with the class means
    theta = theta0 + 0.5 * (data.means / jnp.sqrt(lam)[:, None]) @ jnp.eye(len(PRIORS))
    B = theta.T @ (lam[:, None] * theta)
    O = theta.T @ data.means
    design = jnp.concatenate([jnp.sqrt(lam)[:, None] * theta,          # Kbar Theta
                              data.means / jnp.sqrt(lam)[:, None]], 1)  # D^{-1} Xbar
    results = {"batches": np.array(args.coef_batches)}
    for b in args.coef_batches:
        layout, draws = coefficient_draws(args, b, len(PRIORS), len(PRIORS), seed=b)
        co = discounts.stationary_coefficients(B, O, problem.priors, draws, problem.f,
                                               args.beta1, args.beta2, args.epsilon,
                                               burn=layout[0], tail=layout[2])
        sg = discounts.stationary_coefficients(B, O, problem.priors, draws, problem.f,
                                               args.beta1, args.beta2, args.epsilon,
                                               burn=layout[0], tail=layout[2], sgd=True)
        for k in ("phi", "chi", "sigma", "tau"):
            results[f"{k}_b{b}"] = np.asarray(co[k])
        # beta2 -> 1 limit: E[grad f] / sqrt(E f^2 / b + eps), E f^2 = b * sigma_sgd
        scale = 1 / np.sqrt(np.diag(np.asarray(sg["sigma"])) + args.epsilon)
        results[f"phi1_b{b}"] = np.asarray(sg["phi"]) * scale[None, :]
        results[f"chi1_b{b}"] = np.asarray(sg["chi"]) * scale[None, :]
        if b in args.drift_batches:
            t0 = time.time()
            u = frozen_drift(problem, data, theta, args, b, steps=args.drift_steps,
                             chains=args.drift_chains)
            pred = design @ jnp.concatenate([co["phi"], co["chi"]])
            # drift of the mean logits O = Theta^T Xbar, and of the whole direction
            results[f"measured_b{b}"] = np.asarray(data.means.T @ u)
            results[f"predicted_b{b}"] = np.asarray(data.means.T @ pred)
            results[f"relerr_b{b}"] = float(jnp.linalg.norm(u - pred) / jnp.linalg.norm(pred))
            results[f"relerr_O_b{b}"] = float(jnp.linalg.norm(data.means.T @ (u - pred))
                                              / jnp.linalg.norm(data.means.T @ pred))
            print(f"drift b={b}: relative error of the mean direction "
                  f"{results[f'relerr_b{b}']:.4f}, of its projection on Xbar "
                  f"{results[f'relerr_O_b{b}']:.4f}  [{time.time()-t0:.0f}s]", flush=True)
    np.savez_compressed(args.output_dir / "multiclass_drift.npz", **results)


# --------------------------------------------------------------------------- #
def markers(ax, x, runs, color, every=1, marker="o"):
    mean = runs.mean(0)[::every]
    lo, hi = np.percentile(runs, [10, 90], axis=0)
    ax.errorbar(x[::every], mean, yerr=[np.maximum(mean - lo[::every], 0), np.maximum(hi[::every] - mean, 0)], fmt=marker,
                color=color, ms=5, capsize=2, lw=1)


def plot_batch(args):
    r = dict(np.load(args.output_dir / "multiclass_batch.npz"))
    cols = plt.cm.viridis(np.linspace(0.05, 0.85, len(args.batches)))
    fig, axes = plt.subplots(2, 3, figsize=(20, 10.5), constrained_layout=True)
    for row, method in enumerate(("adam", "sgd")):
        for b, c in zip(args.batches, cols):
            p = f"{method}_b{b}"
            for col, (scale, key) in enumerate(((1, "risk"), (b, "risk"), (1, "error"))):
                ax = axes[row, col]
                start = 1 if col == 1 else 0           # t = 0 is off the log axis
                ax.plot(r[f"{p}_ode_time"][start:] * scale, r[f"{p}_ode_{key}"][start:],
                        color=c, lw=2, label=f"$b={b}$" if col == 0 else None)
                markers(ax, r[f"{p}_time"][start:] * scale, r[f"{p}_{key}"][:, start:], c,
                        every=2)
        name = "Adam" if method == "adam" else "SGD"
        axes[row, 0].set_title(f"{name}: risk vs iterations")
        axes[row, 1].set_title(f"{name}: risk vs samples")
        axes[row, 2].set_title(f"{name}: classification error")
        axes[row, 0].set_xlabel(r"$t=k/D$")
        axes[row, 1].set_xlabel(r"samples per dimension $bt$")
        axes[row, 2].set_xlabel(r"$t=k/D$")
        axes[row, 0].set_ylabel("cross-entropy")
        axes[row, 1].set_ylabel("cross-entropy")
        axes[row, 2].set_ylabel("error")
        axes[row, 1].set_xscale("log")
        axes[row, 0].legend(frameon=False, title="lines: ODE, markers: runs")
    for ax in axes.flat:
        polish_axis(ax)
    save(fig, args.output_dir / "multiclass_batch")


def plot_scaling(args):
    s = dict(np.load(args.output_dir / "multiclass_scaling.npz"))
    dr = dict(np.load(args.output_dir / "multiclass_drift.npz"))
    fig, axes = plt.subplots(1, 4, figsize=(24, 5.6), constrained_layout=True)
    cols = plt.cm.plasma(np.linspace(0.05, 0.75, len(args.scaling_batches)))
    dims = np.array(args.scaling_dimensions)
    for b, c in zip(args.scaling_batches, cols):
        g = np.array([s[f"gap_b{b}_D{d}"] for d in dims])
        for ax, x in ((axes[0], dims), (axes[1], dims / b)):
            ax.errorbar(x, g[:, 0], yerr=2 * g[:, 1], marker="o", color=c, capsize=3,
                        label=f"$b={b}$")
    for ax, xl in ((axes[0], "$D$"), (axes[1], "$D/b$")):
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(xl)
        ax.set_ylabel(r"$\sup_t|\overline{\mathcal{R}}(t)-\mathscr{R}(t)|$")
        ax.legend(frameon=False)
    x = dims / min(args.scaling_batches)
    axes[0].set_title("Mean-risk gap to the ODE ($\\pm2$ s.e.)")
    axes[1].set_title("Same, against $D/b$")
    ref = np.geomspace(dims.min() / max(args.scaling_batches), dims.max(), 10)
    axes[1].plot(ref, 2.0 / ref, "k--", lw=1.4, label=r"$\propto b/D$")
    axes[1].legend(frameon=False)

    ax = axes[2]
    bs = dr["batches"]
    cr = lambda key: np.array([dr[f"{key}_b{b}"] for b in bs])
    phi, phi1 = cr("phi"), cr("phi1")
    chi, chi1 = cr("chi"), cr("chi1")
    ax.plot(bs, np.trace(phi, axis1=1, axis2=2) / phi.shape[1], "o-", color="#1f77b4",
            label=r"$\mathrm{tr}\,\phi_b/C$")
    ax.plot(bs, np.trace(phi1, axis1=1, axis2=2) / phi.shape[1], "--", color="#1f77b4",
            label=r"$\beta_2\to1$ limit")
    ax.plot(bs, -np.trace(chi, axis1=1, axis2=2) / chi.shape[1], "s-", color="#d62728",
            label=r"$-\mathrm{tr}\,\chi_b/C$")
    ax.plot(bs, -np.trace(chi1, axis1=1, axis2=2) / chi.shape[1], "--", color="#d62728")
    ax.plot(bs, np.trace(cr("sigma"), axis1=1, axis2=2) / chi.shape[1], "^-", color="#2ca02c",
            label=r"$\mathrm{tr}\,\hat\sigma_b/C$")
    ax.plot(bs, np.trace(cr("tau"), axis1=1, axis2=2) / chi.shape[1], "v:", color="#2ca02c",
            label=r"$\mathrm{tr}\,\tau_b/C$")
    ax.plot(bs, np.trace(phi[0]) / phi.shape[1] * np.sqrt(bs), "k:", lw=1.2,
            label=r"$\phi_1\sqrt{b}$")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("batch size $b$")
    ax.set_title("Batch coefficients")
    ax.legend(frameon=False, fontsize=11)

    ax = axes[3]
    cols = plt.cm.viridis(np.linspace(0.05, 0.85, len(args.drift_batches)))
    lim = 0
    for b, c in zip(args.drift_batches, cols):
        pred, meas = dr[f"predicted_b{b}"], dr[f"measured_b{b}"]
        ax.plot(pred.ravel(), meas.ravel(), "o", color=c, ms=6,
                label=f"$b={b}$ (rel. err. {dr[f'relerr_O_b{b}']:.1%})")
        lim = max(lim, np.abs(pred).max(), np.abs(meas).max())
    ax.plot([-lim, lim], [-lim, lim], "k--", lw=1.2)
    ax.set_xlabel(r"predicted $\overline{X}^\top(\overline{K}\Theta\phi_b+D^{-1}\overline{X}\chi_b)$")
    ax.set_ylabel(r"measured $\overline{X}^\top\,\mathbb{E}[m/\sqrt{v+\epsilon}]$")
    ax.set_title(f"Direct drift check, $D={args.drift_dimension}$")
    ax.legend(frameon=False)
    for ax in axes:
        polish_axis(ax)
    save(fig, args.output_dir / "multiclass_scaling")


def plot_haar(args):
    r = dict(np.load(args.output_dir / "multiclass_haar.npz"))
    fig, axes = plt.subplots(1, 3, figsize=(20, 5.8), constrained_layout=True)
    colors = {"diagonal": "#1f77b4", "haar": "#d62728"}
    styles = dict(zip(args.haar_batches, ("o", "s", "^")))
    for geometry in ("diagonal", "haar"):
        for b in args.haar_batches:
            p = f"gmm_{geometry}_b{b}"
            for ax, key in ((axes[0], "risk"), (axes[1], "error")):
                ax.plot(r[f"{p}_ode_time"], r[f"{p}_ode_{key}"], color=colors[geometry], lw=2,
                        ls="-" if b == args.haar_batches[0] else "--",
                        label=f"{geometry} $K$, $b={b}$" if key == "risk" else None)
                markers(ax, r[f"{p}_time"], r[f"{p}_{key}"], colors[geometry], 2, styles[b])
    axes[0].set_title(f"5-class mixture, $D={args.haar_dimension}$")
    axes[1].set_title("Classification error")
    axes[0].set_ylabel("cross-entropy")
    axes[1].set_ylabel("error")
    axes[0].legend(frameon=False, title="lines: ODE, markers: runs")
    ax = axes[2]
    cols = plt.cm.viridis(np.linspace(0.1, 0.7, len(args.teacher_batches)))
    for b, c in zip(args.teacher_batches, cols):
        p = f"teacher_b{b}"
        ax.plot(r[f"{p}_ode_time"], r[f"{p}_ode_risk"], color=c, lw=2, label=f"$b={b}$")
        markers(ax, r[f"{p}_time"], r[f"{p}_risk"], c, 2)
    ax.set_title(f"{args.teacher_classes}-class softmax teacher, Haar $K$")
    ax.set_ylabel("cross-entropy")
    ax.legend(frameon=False)
    for ax in axes:
        ax.set_xlabel(r"$t=k/D$")
        polish_axis(ax)
    save(fig, args.output_dir / "multiclass_haar")


def save(fig, path):
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved -> {path}.{{pdf,png}}")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=["batch", "haar", "scaling", "drift", "plot", "all"])
    p.add_argument("--dimension", type=int, default=4000)
    p.add_argument("--batches", type=int, nargs="+", default=[1, 4, 16, 64])
    p.add_argument("--runs", type=int, default=16)
    p.add_argument("--horizon", type=float, default=6.0)
    p.add_argument("--eta", type=float, default=0.5)
    p.add_argument("--sgd-eta", type=float, default=2.0)
    p.add_argument("--beta1", type=float, default=0.9)
    p.add_argument("--beta2", type=float, default=0.95)
    p.add_argument("--epsilon", type=float, default=1e-2)
    p.add_argument("--dt", type=float, default=0.025)
    p.add_argument("--record-per-unit", type=int, default=8)
    p.add_argument("--paths", type=int, default=256)
    p.add_argument("--window", type=int, default=1024)
    p.add_argument("--haar-dimension", type=int, default=2000)
    p.add_argument("--haar-batches", type=int, nargs="+", default=[1, 16])
    p.add_argument("--teacher-classes", type=int, default=4)
    p.add_argument("--teacher-batches", type=int, nargs="+", default=[1, 8, 32])
    p.add_argument("--teacher-eta", type=float, default=0.5)
    p.add_argument("--scaling-dimensions", type=int, nargs="+",
                   default=[500, 1000, 2000, 4000, 8000])
    p.add_argument("--scaling-batches", type=int, nargs="+", default=[1, 8, 64])
    p.add_argument("--coef-batches", type=int, nargs="+",
                   default=[1, 2, 4, 8, 16, 32, 64, 128])
    p.add_argument("--drift-batches", type=int, nargs="+", default=[1, 8, 64])
    p.add_argument("--drift-dimension", type=int, default=2000)
    p.add_argument("--drift-steps", type=int, default=20000)
    p.add_argument("--drift-chains", type=int, default=32)
    p.add_argument("--output-dir", type=Path, default=ROOT / "figs" / "experiments")
    return p.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, fn in (("batch", run_batch), ("haar", run_haar), ("scaling", run_scaling),
                     ("drift", run_drift)):
        if args.command in (name, "all"):
            fn(args)
    if args.command in ("plot", "all"):
        plot_batch(args)
        plot_scaling(args)
        plot_haar(args)


if __name__ == "__main__":
    main()
