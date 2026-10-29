#!/usr/bin/env python3
"""Adam on Gaussian-mixture classification: simulations vs the GMM ODEs.

Data ``x | c ~ N(mu_c, K)`` with shared diagonal ``K`` and ``||mu_c|| = O(1)``;
batch-one streaming Adam with ``eps * K_ii`` stability and step ``eta(k/D)/D``.
All ODE/coefficient code lives in ``gmm.py``; this file only builds scenarios,
runs them and plots.

Scenarios
  binary_iso    means +-mu (||mu|| = 1.5), logistic loss, K = I
  binary_aniso  same with a decaying diagonal K in [0.25, 2]
  softmax3      three components, softmax cross-entropy, decaying diagonal K,
                class weights (0.5, 0.3, 0.2), warmup + cosine learning rate

Subcommands
  main    every scenario for each D: Adam runs + ODE (saved to npz)
  sweep   binary_iso at one D for several (beta1, beta2, eps): Adam, ODE and the
          beta2 -> 1 (preconditioned SGD) ODE; base setting also without diffusion
  drift   frozen-parameter check of the drift (first moment) at several D
  plot    make every figure from the saved npz files
"""
from __future__ import annotations

import argparse
import json
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
import matplotlib.lines
import matplotlib.ticker
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import gmm
from experiments.plot_style import apply_paper_style, polish_axis

apply_paper_style()

SCENARIOS = ("binary_iso", "binary_aniso", "softmax3")
TITLES = {
    "binary_iso": "Binary, isotropic $K$",
    "binary_aniso": "Binary, anisotropic $K$",
    "softmax3": "3-class softmax, anisotropic $K$",
}
# Validated categorical slots (dataviz reference palette, first three = all-pairs safe).
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
SCENARIO_STYLE = {"binary_iso": (BLUE, "o"), "binary_aniso": (ORANGE, "s"),
                  "softmax3": (AQUA, "^")}
INK, MUTED = "#0b0b0b", "#52514e"
RAMP = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281"]   # ordinal blue (D)


# --------------------------------------------------------------------------- #
# Scenarios
# --------------------------------------------------------------------------- #
def spectrum(d, lower=0.25, upper=2.0, decay=4.0):
    """Same decaying profile as ``experiments/diagonal_linreg.py``."""
    u = np.linspace(0.0, 1.0, d)
    raw = (np.exp(-decay * u) - np.exp(-decay)) / (1.0 - np.exp(-decay))
    return lower + (upper - lower) * raw


def cosine_schedule(eta_max, horizon, warmup=0.5, floor=0.1):
    def lr(t):
        warm = jnp.minimum(1.0, t / warmup)
        frac = jnp.clip(t / horizon, 0.0, 1.0)
        return eta_max * warm * (floor + (1.0 - floor) * 0.5 * (1.0 + jnp.cos(jnp.pi * frac)))
    return lr


def build_scenario(name, D, seed, horizon, hp_override=None):
    rng = np.random.default_rng(seed + 1009 * D)
    if name in ("binary_iso", "binary_aniso"):
        cov = np.ones(D) if name == "binary_iso" else spectrum(D)
        u = rng.standard_normal(D)
        u /= np.linalg.norm(u)
        w = rng.standard_normal(D)
        w -= u * (u @ w)
        w /= np.linalg.norm(w)
        mu = 1.5 * u
        means = np.stack([-mu, mu], axis=1)               # class 0: y=-1, class 1: y=+1
        theta0 = (-0.3 * u + 0.6 * w)[:, None]            # starts on the wrong side
        probs = np.array([0.5, 0.5])
        problem = gmm.binary_logistic_problem()
        hp = dict(lr=0.5, beta1=0.9, beta2=0.95, eps=1e-2)
    elif name == "softmax3":
        cov = spectrum(D)
        Q, _ = np.linalg.qr(rng.standard_normal((D, 3)))
        means = np.stack([1.6 * Q[:, 0],
                          1.6 * (0.5 * Q[:, 0] + np.sqrt(0.75) * Q[:, 1]),
                          1.3 * Q[:, 2]], axis=1)
        theta0 = 0.4 * rng.standard_normal((D, 3)) / np.sqrt(D)
        probs = np.array([0.5, 0.3, 0.2])
        problem = gmm.softmax_problem(3, error_samples=20_000)
        hp = dict(lr=cosine_schedule(0.6, horizon), beta1=0.9, beta2=0.95, eps=1e-2)
    else:
        raise ValueError(name)
    if hp_override:
        hp.update(hp_override)
    return dict(problem=problem, cov=jnp.asarray(cov), means=jnp.asarray(means),
                theta0=jnp.asarray(theta0), probs=jnp.asarray(probs), hp=hp)


def observables(name, B, Mbar):
    """Low-dimensional summaries plotted alongside the risk."""
    B, Mbar = np.asarray(B), np.asarray(Mbar)
    if name.startswith("binary"):
        return {"overlap": Mbar[..., 0, 1:2], "B": B[..., 0, 0:1]}
    return {"overlap": np.diagonal(Mbar, axis1=-2, axis2=-1),
            "B": np.diagonal(B, axis1=-2, axis2=-1)}


def run_ode(sc, horizon, dt, seed, variant="adam"):
    hp = sc["hp"]
    return gmm.run_gmm_ode(
        sc["problem"], sc["theta0"], sc["cov"], sc["means"], sc["probs"],
        lr=hp["lr"], beta1=hp["beta1"], beta2=hp["beta2"], eps=hp["eps"],
        T=horizon, dt=dt, key=jax.random.PRNGKey(seed), variant=variant,
        config=gmm.coefficient_config(hp["beta1"], hp["beta2"]))


def run_simulation(sc, D, horizon, runs, records_per_unit, seed):
    hp = sc["hp"]
    record_every = max(1, D // records_per_unit)
    steps = int(round(horizon * D)) // record_every * record_every
    keys = jax.random.split(jax.random.PRNGKey(seed), runs)
    B, Mbar = gmm.run_adam(sc["problem"], sc["theta0"], sc["cov"], sc["means"],
                           sc["probs"], lr=hp["lr"], beta1=hp["beta1"],
                           beta2=hp["beta2"], eps=hp["eps"], num_steps=steps,
                           record_every=record_every, keys=keys)
    risk, err = gmm.risks_from_paths(sc["problem"], B, Mbar, sc["probs"])
    return np.arange(B.shape[1]) * record_every / D, np.asarray(B), np.asarray(Mbar), risk, err


def interp_ode(ode_time, values, time):
    values = np.asarray(values)
    flat = values.reshape(len(ode_time), -1)
    out = np.stack([np.interp(time, ode_time, flat[:, k]) for k in range(flat.shape[1])], -1)
    return out.reshape((len(time),) + values.shape[1:])


# --------------------------------------------------------------------------- #
# Subcommands
# --------------------------------------------------------------------------- #
def cmd_main(args):
    for name in args.scenarios:
        for D in args.dimensions:
            path = args.output_dir / f"gmm_{name}_D{D}.npz"
            if path.exists() and not args.overwrite:
                print(f"skip {path.name}")
                continue
            sc = build_scenario(name, D, args.seed, args.horizon)
            t0 = time.time()
            t_adam, B, Mbar, risk, err = run_simulation(
                sc, D, args.horizon, args.runs, args.records_per_unit, args.seed + 7 * D)
            t1 = time.time()
            ode = run_ode(sc, args.horizon, args.ode_dt, args.seed + 11 * D)
            t2 = time.time()
            obs_a, obs_o = observables(name, B, Mbar), observables(name, ode.B, ode.Mbar)
            result = {
                "time": t_adam, "adam_risk": risk, "adam_error": err,
                "ode_risk": interp_ode(ode.times, ode.risk, t_adam),
                "ode_error": interp_ode(ode.times, ode.error, t_adam),
                "ode_time_fine": ode.times, "ode_risk_fine": ode.risk,
                "D": D,
            }
            for key in obs_a:
                result[f"adam_{key}"] = obs_a[key]
                result[f"ode_{key}"] = interp_ode(ode.times, obs_o[key], t_adam)
            np.savez_compressed(path, **result)
            gap = np.max(np.abs(risk.mean(0) - result["ode_risk"]))
            print(f"{name} D={D}: adam {t1 - t0:.0f}s ode {t2 - t1:.0f}s "
                  f"sup|mean risk - ode| = {gap:.4f}", flush=True)


SWEEP = [
    ("base", dict(beta1=0.9, beta2=0.95, eps=1e-2)),
    ("no momentum, short memory", dict(beta1=0.0, beta2=0.8, eps=1e-3)),
    ("long memory", dict(beta1=0.95, beta2=0.99, eps=1e-2)),
    ("sign-like", dict(beta1=0.5, beta2=0.5, eps=1e-4)),
]


def cmd_sweep(args):
    D = args.sweep_dimension
    settings = args.sweep_settings if args.sweep_settings is not None else range(len(SWEEP))
    for k in settings:
        label, hp = SWEEP[k]
        path = args.output_dir / f"gmm_sweep_{k}{args.sweep_tag}.npz"
        if path.exists() and not args.overwrite:
            print(f"skip {path.name}")
            continue
        sc = build_scenario("binary_iso", D, args.seed, args.horizon, hp_override=hp)
        t_adam, B, Mbar, risk, err = run_simulation(
            sc, D, args.horizon, args.sweep_runs, args.records_per_unit, args.seed + 13 * k)
        result = {"time": t_adam, "adam_risk": risk, "label": label,
                  "hp": json.dumps(hp), "D": D}
        variants = ["adam", "psgd"] + (["no_diffusion"] if k == 0 and not args.sweep_tag else [])
        for variant in variants:
            ode = run_ode(sc, args.horizon, args.ode_dt, args.seed + 17 * k, variant)
            result[f"ode_{variant}"] = interp_ode(ode.times, ode.risk, t_adam)
        np.savez_compressed(path, **result)
        print(f"sweep {label}: " + ", ".join(
            f"{v} gap {np.max(np.abs(risk.mean(0) - result[f'ode_{v}'])):.4f}"
            for v in variants), flush=True)


def cmd_drift(args):
    """Average Adam direction at a frozen Theta vs the predicted drift."""
    name = "binary_aniso"
    for D in args.drift_dimensions:
        path = args.output_dir / f"gmm_drift_D{D}.npz"
        if path.exists() and not args.overwrite:
            print(f"skip {path.name}")
            continue
        sc = build_scenario(name, D, args.seed, args.horizon)
        hp, theta, cov, means, probs = sc["hp"], sc["theta0"], sc["cov"], sc["means"], sc["probs"]
        B = theta.T @ (cov[:, None] * theta)
        Mbar = theta.T @ means
        cfg = gmm.coefficient_config(hp["beta1"], hp["beta2"], num_paths=1024)
        noise = gmm.draw_base_noise(jax.random.PRNGKey(args.seed + 3), 1, cfg)
        co = gmm.adam_coefficients_from_noise(B, Mbar, probs, noise, problem=sc["problem"],
                                              beta1=hp["beta1"], beta2=hp["beta2"],
                                              eps=hp["eps"], config=cfg)
        cp = gmm.psgd_coefficients(B, Mbar, probs, noise, problem=sc["problem"],
                                   eps=hp["eps"], config=cfg)
        t0 = time.time()
        dirs = np.asarray(gmm.frozen_adam_mean_update(
            sc["problem"], theta, cov, means, probs, beta1=hp["beta1"],
            beta2=hp["beta2"], eps=hp["eps"], num_steps=args.drift_steps,
            burn_in=500, key=jax.random.PRNGKey(args.seed + D), num_chains=8))
        preds = {
            "adam": gmm.predicted_drift(theta, cov, means, co),
            "no_mean": gmm.predicted_drift(theta, cov, means, (co[0], 0 * co[1], co[2])),
            "psgd": gmm.predicted_drift(theta, cov, means, cp),
        }
        np.savez_compressed(path, measured=dirs, theta=np.asarray(theta),
                            cov=np.asarray(cov), means=np.asarray(means),
                            **{f"pred_{k}": np.asarray(v) for k, v in preds.items()})
        print(f"drift D={D}: {time.time() - t0:.0f}s", flush=True)


# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #
def load(args, name, D):
    return dict(np.load(args.output_dir / f"gmm_{name}_D{D}.npz"))


def central(x, alpha=0.2):
    return np.quantile(x, alpha / 2, axis=0), np.quantile(x, 1 - alpha / 2, axis=0)


def save(fig, args, stem):
    fig.savefig(args.output_dir / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(args.output_dir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def marker_times(time):
    """Dense early (fast phase), sparse late: indices for marker overlays."""
    grid = np.concatenate([np.arange(0.125, 1.0001, 0.125), np.arange(1.5, time[-1] + 1e-9, 0.5)])
    return np.unique(np.clip(np.searchsorted(time, grid - 1e-9), 0, len(time) - 1))


def adam_markers(ax, time, samples, idx, color, marker, label=None, ms=6.5):
    lo, hi = central(samples)
    mean = samples.mean(0)
    ax.errorbar(time[idx], mean[idx], yerr=[mean[idx] - lo[idx], hi[idx] - mean[idx]],
                fmt=marker, ms=ms, color=color, mec="white", mew=0.9, elinewidth=1.3,
                capsize=0, zorder=3, label=label)


def log_dimension_axis(ax, dims):
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks(dims)
    ax.set_xticklabels([str(d) for d in dims])
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())


def plot_parity(args):
    D = max(args.dimensions)
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 6.0), constrained_layout=True)
    hi_all = []
    for name in args.scenarios:
        r = load(args, name, D)
        color, marker = SCENARIO_STYLE[name]
        idx = marker_times(r["time"])
        for ax, key in zip(axes, ("risk", "error")):
            pred, obs = r[f"ode_{key}"][idx], r[f"adam_{key}"][:, idx]
            lo, hi = central(obs)
            mean = obs.mean(0)
            ax.errorbar(pred, mean, yerr=[mean - lo, hi - mean], fmt=marker, ms=8,
                        color=color, mec="white", mew=1.0, elinewidth=1.4, capsize=0,
                        label=TITLES[name])
            if key == "risk":
                hi_all.append(max(pred.max(), hi.max()))
    for ax, lim in zip(axes, (1.05 * max(hi_all), 0.8)):
        ax.plot((0, lim), (0, lim), color=MUTED, lw=1.5, zorder=0)
        ax.set_xlim(0, lim)
        ax.set_ylim(0, lim)
    for ax, what in zip(axes, ("risk", "classification error")):
        polish_axis(ax)
        ax.set_aspect("equal")
        ax.set_xlabel(f"ODE-predicted {what}")
        ax.set_ylabel(f"Adam {what}")
    axes[0].set_title("Population risk")
    axes[1].set_title("Classification error")
    axes[0].legend(frameon=False, loc="upper left", fontsize=13)
    fig.suptitle(f"Adam vs. ODE prediction at matched times ($D={D}$, 24 runs; "
                 "bars: central 80%)", fontsize=17)
    save(fig, args, "gmm_risk_vs_predicted")


def plot_risk_curves(args):
    dims = sorted(args.dimensions)
    shown = [dims[0], dims[len(dims) // 2], dims[-1]]
    shades = dict(zip(shown, (RAMP[0], RAMP[2], RAMP[4])))
    n = len(args.scenarios)
    fig, axes = plt.subplots(2, n, figsize=(6.0 * n, 8.2), constrained_layout=True,
                             squeeze=False, sharex=True,
                             gridspec_kw={"height_ratios": [1.6, 1.0]})
    for col, name in enumerate(args.scenarios):
        r = load(args, name, dims[-1])
        ax = axes[0, col]
        ax.plot(r["ode_time_fine"], r["ode_risk_fine"], color=INK, lw=2.0, label="ODE",
                zorder=2)
        adam_markers(ax, r["time"], r["adam_risk"], marker_times(r["time"]), BLUE, "o",
                     label=f"Adam, $D={dims[-1]}$ (mean, central 80%)")
        polish_axis(ax)
        ax.set_title(TITLES[name])
        ax = axes[1, col]
        for D in shown:
            rD = load(args, name, D)
            lo, hi = central(rD["adam_risk"] - rD["ode_risk"][None])
            ax.fill_between(rD["time"], lo, hi, color=shades[D], alpha=0.75, lw=0,
                            label=f"$D={D}$")
        ax.axhline(0.0, color=INK, lw=1.2)
        polish_axis(ax)
        ax.set_xlabel(r"Rescaled time $t=k/D$")
    axes[0, 0].set_ylabel("Population risk")
    axes[1, 0].set_ylabel("Adam $-$ ODE\n(central 80% of runs)")
    axes[0, 0].legend(frameon=False, loc="upper right", fontsize=12.5)
    axes[1, 0].legend(frameon=False, loc="upper right", fontsize=12.5, ncol=3)
    save(fig, args, "gmm_risk_curves")


def plot_order_parameters(args):
    D = max(args.dimensions)
    rows = [("overlap", r"mean logit $O_{rc}=\theta_r^{\top}\bar{x}_c$"),
            ("B", r"$B_{rr}=\theta_r^{\top}K\theta_r$"),
            ("error", "classification error")]
    fig, axes = plt.subplots(len(rows), len(args.scenarios),
                             figsize=(5.6 * len(args.scenarios), 3.6 * len(rows)),
                             constrained_layout=True, sharex=True, squeeze=False)
    colors, markers = [BLUE, ORANGE, AQUA], ["o", "s", "^"]
    for col, name in enumerate(args.scenarios):
        r = load(args, name, D)
        idx = marker_times(r["time"])
        for row, (key, label) in enumerate(rows):
            ax = axes[row, col]
            adam, ode = r[f"adam_{key}"], r[f"ode_{key}"]
            if adam.ndim == 2:
                adam, ode = adam[..., None], ode[..., None]
            for k in range(adam.shape[-1]):
                ax.plot(r["time"], ode[:, k], color=colors[k], lw=2.0, zorder=2)
                adam_markers(ax, r["time"], adam[..., k], idx, colors[k], markers[k], ms=5.5)
            polish_axis(ax)
            if col == 0:
                ax.set_ylabel(label, fontsize=15)
            if row == 0:
                ax.set_title(TITLES[name])
            if row == len(rows) - 1:
                ax.set_xlabel(r"Rescaled time $t=k/D$")
        if name == "softmax3":
            handles = [matplotlib.lines.Line2D([], [], color=colors[k], marker=markers[k],
                                               lw=2.0, mec="white", label=f"class {k}")
                       for k in range(3)]
            axes[0, col].legend(handles=handles, frameon=False, fontsize=12, loc="lower right")
    axes[0, 0].text(0.97, 0.05, r"binary: $O_{1}=\theta^{\top}\bar{x}_{+}$",
                    transform=axes[0, 0].transAxes, ha="right", fontsize=12, color=MUTED)
    fig.suptitle(f"Order parameters, $D={D}$: ODE (lines) vs Adam (markers: mean, "
                 "bars: central 80%)", fontsize=17)
    save(fig, args, "gmm_order_parameters")


def plot_dimension_scaling(args):
    fig, ax = plt.subplots(figsize=(7.0, 5.4), constrained_layout=True)
    dims = np.array(sorted(args.dimensions))
    for name in args.scenarios:
        color, marker = SCENARIO_STYLE[name]
        stats = []
        for D in dims:
            r = load(args, name, D)
            sup = np.max(np.abs(r["adam_risk"] - r["ode_risk"][None]), axis=1)
            stats.append(np.quantile(sup, (0.25, 0.5, 0.75)))
        q1, med, q3 = np.array(stats).T
        ax.errorbar(dims, med, yerr=[med - q1, q3 - med], color=color, marker=marker,
                    ms=8, lw=2.0, capsize=0, mec="white", mew=1.0, label=TITLES[name])
    ref = 1.25 * med[0] * np.sqrt(dims[0] / dims)
    ax.plot(dims, ref, color=MUTED, lw=1.5, ls=(0, (4, 3)), label=r"$\propto D^{-1/2}$")
    log_dimension_axis(ax, dims)
    polish_axis(ax)
    ax.set_xlabel("Dimension $D$")
    ax.set_ylabel(r"$\sup_{t\leq 6}\,|\mathcal{R}(\theta_{\lfloor tD\rfloor})-\mathscr{R}(t)|$")
    ax.set_title("Per-run sup error (median, IQR of 24 runs)", fontsize=16)
    ax.legend(frameon=False, fontsize=12, loc="lower left")
    save(fig, args, "gmm_dimension_scaling")


def plot_sweep(args):
    files = [args.output_dir / f"gmm_sweep_{k}.npz" for k in range(len(SWEEP))]
    files = [p for p in files if p.exists()]
    if not files:
        return
    fig, axes = plt.subplots(2, len(files), figsize=(5.2 * len(files), 7.6),
                             constrained_layout=True, squeeze=False, sharex=True,
                             gridspec_kw={"height_ratios": [1.5, 1.0]})
    for col, path in enumerate(files):
        r = dict(np.load(path))
        hp = json.loads(str(r["hp"]))
        t, risk = r["time"], r["adam_risk"]
        ax = axes[0, col]
        ax.plot(t, r["ode_adam"], color=INK, lw=2.0, label="ODE (this note)", zorder=2)
        ax.plot(t, r["ode_psgd"], color=ORANGE, lw=2.0, ls=(0, (5, 2)),
                label=r"$\beta_2\to1$ approx. (precond. SGD)", zorder=2)
        if "ode_no_diffusion" in r:
            ax.plot(t, r["ode_no_diffusion"], color=AQUA, lw=2.0, ls=(0, (1, 1.5)),
                    label=r"ODE without the $\eta^2\Sigma$ term", zorder=2)
        adam_markers(ax, t, risk, marker_times(t), BLUE, "o", label="Adam (mean, central 80%)",
                     ms=5.5)
        polish_axis(ax)
        ax.set_title(f"$\\beta_1={hp['beta1']},\\ \\beta_2={hp['beta2']},\\ "
                     f"\\epsilon={hp['eps']:g}$", fontsize=16)
        ax = axes[1, col]
        mean = risk.mean(0)
        se = risk.std(0, ddof=1) / np.sqrt(risk.shape[0])
        ax.fill_between(t, -2 * se, 2 * se, color=RAMP[0], alpha=0.7, lw=0,
                        label=r"$\pm2$ s.e. of Adam mean")
        ax.plot(t, r["ode_adam"] - mean, color=INK, lw=2.0, label="ODE")
        ax.plot(t, r["ode_psgd"] - mean, color=ORANGE, lw=2.0, ls=(0, (5, 2)),
                label=r"$\beta_2\to1$ approx.")
        big = args.output_dir / f"gmm_sweep_{col}_D8000.npz"
        if big.exists():
            rb = dict(np.load(big))
            ax.plot(rb["time"], rb["ode_adam"] - rb["adam_risk"].mean(0), color=MUTED,
                    lw=1.6, label=f"ODE, $D={int(rb['D'])}$")
        ax.axhline(0.0, color=MUTED, lw=0.8)
        polish_axis(ax)
        ax.set_xlabel(r"Rescaled time $t=k/D$")
        lim = 1.15 * max(np.max(np.abs(r["ode_psgd"] - mean)), np.max(np.abs(r["ode_adam"] - mean)))
        ax.set_ylim(-lim, lim)
        if big.exists() or col == 0:
            ax.legend(frameon=False, fontsize=10.5, loc="lower right")
    axes[0, 0].set_ylabel("Population risk")
    axes[1, 0].set_ylabel("model $-$ Adam mean")
    axes[0, 0].legend(frameon=False, fontsize=11, loc="upper right")
    D = int(np.load(files[0])["D"])
    fig.suptitle(f"Binary isotropic GMM, $D={D}$, 16 runs: hyperparameters and baselines",
                 fontsize=17)
    save(fig, args, "gmm_sweep")


def plot_drift(args):
    files = sorted(args.output_dir.glob("gmm_drift_D*.npz"),
                   key=lambda p: int(p.stem.split("D")[-1]))
    if not files:
        return
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.8), constrained_layout=True)
    r = dict(np.load(files[0]))
    D0 = r["means"].shape[0]
    meas = r["measured"].mean(0)[:, 0]
    scale = np.sqrt(D0)
    ax = axes[0]
    for key, color, marker, label in [
            ("no_mean", ORANGE, "s", r"prediction without $\mathbf{D}^{-1}\bar{X}\chi$"),
            ("adam", BLUE, "o", "full prediction")]:
        ax.scatter(scale * r[f"pred_{key}"][:, 0], scale * meas, s=12, color=color,
                   marker=marker, alpha=0.6, lw=0, label=label)
    lim = np.max(np.abs(scale * meas)) * 1.05
    ax.plot([-lim, lim], [-lim, lim], color=MUTED, lw=1.5, zorder=0)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    polish_axis(ax)
    ax.set_xlabel(r"predicted $\sqrt{D}\,[\overline{K}\theta\phi+\mathbf{D}^{-1}\bar{X}\chi]_i$")
    ax.set_ylabel(r"measured $\sqrt{D}\,\mathbb{E}[m_i/\sqrt{v_i+\epsilon_i}]$")
    ax.set_title(f"Per-coordinate drift at frozen $\\theta$, $D={D0}$")
    ax.legend(frameon=False, fontsize=12, loc="upper left", markerscale=2)

    ax = axes[1]
    dims, rel, err = [], {k: [] for k in ("adam", "psgd")}, []
    for path in files:
        r = dict(np.load(path))
        mu = r["means"][:, 1]
        dims.append(r["means"].shape[0])
        proj = r["measured"][:, :, 0] @ mu
        m_mean, m_se = proj.mean(), proj.std(ddof=1) / np.sqrt(len(proj))
        err.append(m_se / abs(m_mean))
        for k in rel:
            rel[k].append(abs(r[f"pred_{k}"][:, 0] @ mu - m_mean) / abs(m_mean))
    dims = np.array(dims)
    ax.plot(dims, rel["adam"], color=BLUE, marker="o", ms=8, lw=2.0, label="full prediction")
    ax.plot(dims, rel["psgd"], color=ORANGE, marker="s", ms=8, lw=2.0, ls=(0, (5, 2)),
            label=r"$\beta_2\to1$ approx.")
    ax.plot(dims, 2 * np.array(err), color=MUTED, lw=1.2, label="2 s.e. (Monte Carlo)")
    ax.plot(dims, rel["adam"][0] * dims[0] / dims, color=MUTED, lw=1.5, ls=(0, (4, 3)),
            label=r"$\propto D^{-1}$")
    log_dimension_axis(ax, dims)
    polish_axis(ax)
    ax.set_xlabel("Dimension $D$")
    ax.set_ylabel(r"relative error of $\frac{d}{dt}\,\theta^{\top}\bar{x}_{+}$")
    ax.set_title("Drift along the class mean")
    ax.legend(frameon=False, fontsize=12, loc="center left", bbox_to_anchor=(0.0, 0.33))
    save(fig, args, "gmm_drift_check")


def cmd_table(args):
    """Print the numbers quoted in paper/gmm_adam.tex."""
    print("scenario & D & sup_t|mean Adam - ODE| & median per-run sup error & "
          "max_t run std")
    for name in args.scenarios:
        for D in sorted(args.dimensions):
            path = args.output_dir / f"gmm_{name}_D{D}.npz"
            if not path.exists():
                continue
            r = dict(np.load(path))
            risk, ode = r["adam_risk"], r["ode_risk"]
            mean_gap = np.max(np.abs(risk.mean(0) - ode))
            per_run = np.median(np.max(np.abs(risk - ode[None]), axis=1))
            spread = np.max(risk.std(0))
            print(f"{name} & {D} & {mean_gap:.4f} & {per_run:.4f} & {spread:.4f}")
    for path in sorted(args.output_dir.glob("gmm_sweep_*.npz")):
        r = dict(np.load(path))
        mean = r["adam_risk"].mean(0)
        late = r["time"] >= 1.5
        gaps = {k[4:]: (np.max(np.abs(mean - r[k])), np.max(np.abs(mean - r[k])[late]))
                for k in r if k.startswith("ode_")}
        print(path.stem, str(r["label"]), int(r["D"]),
              {k: f"all {a:.4f} / t>=1.5 {b:.4f}" for k, (a, b) in gaps.items()})
    for path in sorted(args.output_dir.glob("gmm_drift_D*.npz"),
                       key=lambda p: int(p.stem.split("D")[-1])):
        r = dict(np.load(path))
        mu = r["means"][:, 1]
        proj = r["measured"][:, :, 0] @ mu
        m, se = proj.mean(), proj.std(ddof=1) / np.sqrt(len(proj))
        meas = r["measured"].mean(0)
        se_coord = r["measured"].std(0, ddof=1) / np.sqrt(r["measured"].shape[0])
        out = []
        for k in ("adam", "psgd", "no_mean"):
            pred = r[f"pred_{k}"]
            rel_proj = (pred[:, 0] @ mu - m) / abs(m)
            excess = np.sqrt(max(np.sum((meas - pred) ** 2) - np.sum(se_coord ** 2), 0.0))
            out.append(f"{k}: proj rel {rel_proj:+.4f}, vec rel {excess / np.linalg.norm(pred):.4f}")
        print(path.stem, f"measured <d,mu+> = {m:.4f} +- {se:.4f};", "; ".join(out))


def cmd_plot(args):
    plot_parity(args)
    plot_risk_curves(args)
    plot_order_parameters(args)
    plot_dimension_scaling(args)
    plot_sweep(args)
    plot_drift(args)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["main", "sweep", "drift", "plot", "table", "all"])
    parser.add_argument("--scenarios", nargs="+", default=list(SCENARIOS), choices=SCENARIOS)
    parser.add_argument("--dimensions", type=int, nargs="+",
                        default=[250, 500, 1000, 2000, 4000])
    parser.add_argument("--runs", type=int, default=24)
    parser.add_argument("--horizon", type=float, default=6.0)
    parser.add_argument("--records-per-unit", type=int, default=20)
    parser.add_argument("--ode-dt", type=float, default=0.02)
    parser.add_argument("--sweep-dimension", type=int, default=2000)
    parser.add_argument("--sweep-runs", type=int, default=16)
    parser.add_argument("--sweep-settings", type=int, nargs="+", default=None,
                        help="indices into SWEEP (default: all)")
    parser.add_argument("--sweep-tag", default="",
                        help="suffix for the sweep npz files (e.g. _D8000)")
    parser.add_argument("--drift-dimensions", type=int, nargs="+", default=[1000, 4000])
    parser.add_argument("--drift-steps", type=int, default=150_000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "figs" / "experiments")
    return parser.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    commands = {"main": cmd_main, "sweep": cmd_sweep, "drift": cmd_drift, "plot": cmd_plot,
                "table": cmd_table}
    for name in (["main", "sweep", "drift", "plot"] if args.command == "all" else [args.command]):
        commands[name](args)
    with (args.output_dir / "gmm_config.json").open("w") as handle:
        json.dump({k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                  handle, indent=2)


if __name__ == "__main__":
    main()
