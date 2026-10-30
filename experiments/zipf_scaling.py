#!/usr/bin/env python3
"""Adam scaling laws for Zipf-imbalanced multiclass Gaussian mixtures.

Model (experiments/zipf_adam_plan.md): ``C`` classes with priors
``pi_c ~ c^{-alpha}``, means ``r q_c`` with ``q_c`` orthonormal, ``K = Id``,
softmax cross-entropy with ``M = C`` logits and no bias, ``Theta_0 = 0.1 N(0, 1/D)``.
For ``K = Id`` the mode ODEs aggregate exactly (``dynamics.isotropic_modes``),
and the coefficients come from ``discounts.softmax_coefficients``, which pools
ordinary and forced-class events so that rare classes are resolved.

Subcommands: ``selftest | timings | validate | rates | timescales | exponents |
batch | noise | plot``. Results go to ``figs/experiments/zipf/`` (``.npz`` and
``.pdf``); the log is ``experiments/zipf/README.md``.
"""
from __future__ import annotations

import argparse
import functools
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
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import discounts
import dynamics
import problems
import simulate
from experiments.plot_style import apply_paper_style, polish_axis

apply_paper_style()

OUT = ROOT / "figs" / "experiments" / "zipf"
MEAN_NORM = 5.0          # r: Bayes error <= 0.7%, no-bias rule <= 1.4% per class at C = 128
SAFETY = 0.5             # ODE step factor: per-class losses within 0.13% of safety 0.1 (C = 32)


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
def zipf_priors(classes, alpha):
    pi = np.arange(1, classes + 1, dtype=float) ** -alpha
    return pi / pi.sum()


def zipf_model(classes, alpha, dimension, *, r=MEAN_NORM, seed=0):
    """Priors, isotropic mixture data and ``Theta_0`` (``D x C``)."""
    k_q, k_t = jax.random.split(jax.random.PRNGKey(seed + 7919 * classes + dimension))
    Q, _ = jnp.linalg.qr(jax.random.normal(k_q, (dimension, classes)))
    theta0 = 0.1 * jax.random.normal(k_t, (dimension, classes)) / jnp.sqrt(dimension)
    data = simulate.MixtureData(jnp.ones(dimension), r * Q, jnp.zeros((dimension, 0)))
    return jnp.asarray(zipf_priors(classes, alpha)), data, theta0


@functools.lru_cache(maxsize=None)
def moment_fn(mode):
    """Path-free SGD / PSGD coefficients (``discounts.softmax_moment_coefficients``)."""
    def fn(B, O, aux):
        priors, draws, eps, batch = aux
        return discounts.softmax_moment_coefficients(B, O, priors, draws, eps, batch,
                                                     mode=mode)
    return fn


def moment_draws(classes, samples=2 ** 15, seed=0):
    return jax.random.normal(jax.random.PRNGKey(seed + 31), (samples, classes))


@functools.lru_cache(maxsize=None)
def coefficient_fn(layout, mode, stein):
    """A hashable coefficient callable for ``dynamics.run_mode_ode_logtime``."""
    def fn(B, O, aux):
        priors, plan, beta1, beta2, eps = aux
        return discounts.softmax_coefficients(B, O, priors, plan, beta1, beta2, eps,
                                              layout=layout, mode=mode, stein=stein)
    return fn


def make_plan(priors, beta1, beta2, batch, *, seed=0, plain_batches=None,
              window=None, min_events=1000, min_chunks=1):
    """Coefficient plan. By default ``max(2^14, 2^18 / b)`` ordinary batches (a
    fixed number of samples), and long ``beta2`` memories get longer windows so
    that the burn-in is amortized."""
    if plain_batches is None:
        plain_batches = max(2 ** 14, 2 ** 18 // batch)
    if window is None:
        window = 1024 if beta2 <= 0.99 else 8192
    return discounts.softmax_plan(jax.random.PRNGKey(seed), priors, beta1, beta2,
                                  batch=batch, plain_batches=plain_batches,
                                  window=window, min_events=min_events,
                                  min_chunks=min_chunks)


def solve(priors, theta0, means, times, *, lr, beta1, beta2, eps, batch, mode="adam",
          stein=False, plan=None, safety=SAFETY, dt_max=np.inf, seed=0):
    """Collapsed isotropic ODE on a time grid. ``mode``: adam | sgd | psgd (the
    last two use the path-free estimator)."""
    modes, initial = dynamics.isotropic_modes(theta0, jnp.zeros((theta0.shape[0], 0)),
                                              means)
    if mode == "adam":
        if plan is None:
            plan = make_plan(priors, beta1, beta2, batch, seed=seed)
        layout, arrays = plan
        fn, aux = coefficient_fn(layout, mode, stein), (priors, arrays, beta1, beta2, eps)
    else:
        fn = moment_fn(mode)
        aux = (priors, moment_draws(len(priors), seed=seed), eps, float(batch))
    return dynamics.run_mode_ode_logtime(fn, aux, modes, initial, lr, times,
                                         dimension=theta0.shape[0], safety=safety,
                                         dt_max=dt_max)


def simulate_runs(priors, data, theta0, record_steps, *, lr, beta1, beta2, eps, batch,
                  runs, sgd=False, seed=0):
    """Adam (or SGD) runs recorded at ``record_steps``; ``lr`` in ODE units."""
    problem = problems.make_gmm_softmax(np.asarray(priors))
    d = theta0.shape[0]
    if sgd:
        sim = simulate.build_mixture_sgd(problem, data, lr / d, batch=batch)
    else:
        sim = simulate.build_mixture_adam(problem, data, lr / d, beta1=beta1,
                                          beta2=beta2, eps=eps, batch=batch)
    keys = jax.random.split(jax.random.PRNGKey(10_000 + seed), runs)
    return simulate.run_mixture_at_many(sim, data, theta0, record_steps, keys=keys)


def class_metrics(B, O, *, samples=2 ** 14):
    """Per-class losses and errors for stacked ``(..., C, C)`` order parameters."""
    flat_B, flat_O = B.reshape((-1,) + B.shape[-2:]), O.reshape((-1,) + O.shape[-2:])
    fn = jax.jit(lambda b, o: problems.softmax_class_metrics(b, o, samples=samples))
    loss, err = jax.lax.map(lambda bo: fn(*bo), (flat_B, flat_O))
    shape = B.shape[:-2] + (B.shape[-1],)
    return np.asarray(loss).reshape(shape), np.asarray(err).reshape(shape)


def log_grid(t_min, t_max, per_decade=12):
    n = int(np.ceil(np.log10(t_max / t_min) * per_decade)) + 1
    return np.concatenate([[0.0], np.geomspace(t_min, t_max, n)])


def first_crossing(times, values, level):
    """First time ``values <= level`` (log-linear interpolation; NaN if never)."""
    times, values = np.asarray(times), np.asarray(values)
    below = np.flatnonzero(values <= level)
    if not len(below):
        return np.nan
    k = below[0]
    if k == 0:
        return times[0]
    t0, t1 = times[k - 1], times[k]
    w = (values[k - 1] - level) / max(values[k - 1] - values[k], 1e-300)
    if t0 > 0:
        return float(np.exp(np.log(t0) + np.clip(w, 0, 1) * (np.log(t1) - np.log(t0))))
    return float(t0 + np.clip(w, 0, 1) * (t1 - t0))


def learning_times(times, loss, err):
    """Per-class ``t_c`` (``ell_c <= ell_c(0)/2``) and ``t_c^err`` (``err_c <= 1/2``)
    from ``(T, C)`` paths."""
    loss, err = np.asarray(loss), np.asarray(err)
    t_loss = np.array([first_crossing(times, loss[:, c], loss[0, c] / 2)
                       for c in range(loss.shape[1])])
    t_err = np.array([first_crossing(times, err[:, c], 0.5) for c in range(err.shape[1])])
    return t_loss, t_err


# --------------------------------------------------------------------------- #
# Theory: per-class rates at Theta = 0 (p = 1/C on every input)
# --------------------------------------------------------------------------- #
def kappa_beta(beta1, beta2):
    """``b pi_c = kappa_beta (1 - beta2)`` equates the rare-event and PSGD rates."""
    return (np.pi / 2) * ((1 - beta1 / np.sqrt(beta2)) / (1 - beta1)) ** 2


def predicted_rates(priors, beta1, beta2, eps, batch, r=MEAN_NORM):
    """``|dO_cc/dt| / eta`` at ``Theta = 0`` in the regimes of the plan.

    * SGD: ``r^2 pi_c (1 - 1/C)``.
    * PSGD (``beta2 -> 1``): ``r^2 pi_c (1 - 1/C) / sqrt(E f_c^2 / b + eps)``
      with ``E f_c^2 = pi_c (1 - 1/C)^2 + (1 - pi_c) / C^2`` (class-c events plus
      the softmax background); ``~ r^2 sqrt(b pi_c)`` in the window.
    * Rare events (a class-c sample dominates its own denominator, the update is
      sign-like): ``r^2 b pi_c c_beta sqrt(2/pi) / (1 - beta1/sqrt(beta2))``.
    """
    pi = np.asarray(priors)
    C = len(pi)
    mean_f = 1 - 1 / C
    second = pi * mean_f ** 2 + (1 - pi) / C ** 2
    c_beta = (1 - beta1) / np.sqrt(1 - beta2)
    return {
        "sgd": r ** 2 * pi * mean_f,
        "psgd": r ** 2 * pi * mean_f / np.sqrt(second / batch + eps),
        "rare": (r ** 2 * batch * pi * c_beta * np.sqrt(2 / np.pi)
                 / (1 - beta1 / np.sqrt(beta2))),
        "kink_beta2": kappa_beta(beta1, beta2) * (1 - beta2) / batch,
        "kink_eps": batch * eps,
        "kink_background": 1 / C ** 2,
    }


def frozen_rate(co, O, r=MEAN_NORM):
    """``dO_cc/dt / eta = -(phi^T O + r^2 chi^T)_cc`` (``K = Id``, orthonormal means)."""
    return -np.diag(np.asarray(co["phi"]).T @ np.asarray(O) + r ** 2 * np.asarray(co["chi"]).T)


# --------------------------------------------------------------------------- #
# selftest and timings (Phase 0)
# --------------------------------------------------------------------------- #
def run_selftest(args):
    sys.path.insert(0, str(ROOT / "tests"))
    import test_zipf_coefficients as tests
    ok = tests.main()
    if not ok:
        raise SystemExit(1)


def run_timings(args):
    rows = []
    for classes in args.timing_classes:
        priors = jnp.asarray(zipf_priors(classes, 2.0))
        for beta2 in args.timing_beta2:
            for batch in args.timing_batches:
                t0 = time.time()
                layout, plan = make_plan(priors, 0.9, beta2, batch)
                t_plan = time.time() - t0
                B = 0.3 * jnp.eye(classes) + 0.01
                O = 2.0 * jnp.eye(classes)
                fn = lambda: discounts.softmax_coefficients(B, O, priors, plan, 0.9, beta2,
                                                            1e-4, layout=layout)
                t0 = time.time()
                jax.block_until_ready(fn())
                t_compile = time.time() - t0
                t0 = time.time()
                for _ in range(3):
                    jax.block_until_ready(fn())
                t_eval = (time.time() - t0) / 3
                rare = int(layout.forced_paths > 0) * layout.events
                rows.append(dict(C=classes, beta2=beta2, b=batch, plan_s=t_plan,
                                 compile_s=t_compile, eval_s=t_eval, rare=rare,
                                 forced_paths=layout.forced_paths,
                                 plain_paths=layout.plain_paths))
                print(f"C={classes:4d} beta2={beta2:5.3f} b={batch:3d}: eval {t_eval*1e3:8.1f} ms"
                      f" (compile {t_compile:5.1f}s, plan {t_plan:4.1f}s), rare classes {rare},"
                      f" forced paths {layout.forced_paths}", flush=True)
    # one full ODE run per C (beta2 = 0.99, b = 1) to time the integrator
    for classes in args.timing_classes:
        priors, data, theta0 = zipf_model(classes, 2.0, 500)
        times = log_grid(0.01, args.timing_horizon)
        t0 = time.time()
        out = solve(priors, theta0, data.means, times, lr=0.1, beta1=0.9, beta2=0.99,
                    eps=1e-8, batch=1)
        jax.block_until_ready(out["B"])
        rows.append(dict(C=classes, ode_s=time.time() - t0,
                         ode_steps=int(np.asarray(out["steps"]).sum())))
        print(f"ODE C={classes}: {rows[-1]['ode_s']:.1f}s for t <= {args.timing_horizon}"
              f" ({rows[-1]['ode_steps']} Heun steps)", flush=True)
    (OUT / "timings.json").write_text(json.dumps(rows, indent=1))


def run_precision(args):
    """Per-class events and batch-means relative standard errors of the diagonals
    of ``chi``, ``phi``, ``sigma`` at two states: ``Theta_0`` and a mid-training
    state ``O = 2 Id``, ``B = 0.3 Id + 0.01``."""
    rows = {}
    for classes in args.precision_classes:
        priors = jnp.asarray(zipf_priors(classes, 2.0))
        _, data, theta0 = zipf_model(classes, 2.0, 2000)
        states = {"init": (theta0.T @ theta0, theta0.T @ data.means),
                  "mid": (0.3 * jnp.eye(classes) + 0.01, 2.0 * jnp.eye(classes))}
        for beta2 in (0.95, 0.99):
            for batch in (1, 16):
                layout, plan = make_plan(priors, 0.9, beta2, batch, min_chunks=16)
                events = np.asarray(plan.counts).sum(0)
                for name, (B, O) in states.items():
                    est, se = discounts.softmax_standard_errors(
                        B, O, priors, plan, 0.9, beta2, 1e-8, layout=layout)
                    rel = np.asarray(se / jnp.abs(est))
                    key = f"C{classes}_b2{beta2}_b{batch}_{name}"
                    rows[key] = {"events_min": int(events.min()),
                                 "rel_se_max": rel.max(axis=1).tolist(),
                                 "rel_se_rarest": rel[:, -1].tolist()}
                    print(f"{key:28s} min events {events.min():6d}; max rel s.e. "
                          f"chi {rel[0].max():.3f} phi {rel[1].max():.3f} sigma {rel[2].max():.3f}"
                          f" (rarest class: {rel[0, -1]:.3f} {rel[1, -1]:.3f} {rel[2, -1]:.3f})",
                          flush=True)
    (OUT / "precision.json").write_text(json.dumps(rows, indent=1))


# --------------------------------------------------------------------------- #
# Learning-rate tuning and E0 (validate)
# --------------------------------------------------------------------------- #
def total(loss, priors):
    return np.asarray(loss) @ np.asarray(priors)


def run_tune(args):
    """eta grid for Adam and SGD on the reference problem (ODE only)."""
    priors, data, theta0 = zipf_model(args.tune_classes, args.tune_alpha, 4000)
    times = log_grid(0.01, args.tune_horizon)
    results, chosen = {"time": times}, {}
    for method, grid in (("adam", args.adam_etas), ("sgd", args.sgd_etas)):
        finals = []
        for eta in grid:
            t0 = time.time()
            res = solve(priors, theta0, data.means, times, lr=eta, beta1=args.beta1,
                        beta2=args.beta2, eps=args.eps, batch=1, mode=method)
            loss, _ = class_metrics(res["B"], res["O"])
            risk = total(loss, priors)
            results[f"{method}_{eta:g}"] = risk
            finals.append(np.interp(np.log(args.tune_target), np.log(times[1:]), risk[1:]))
            print(f"tune {method} eta={eta:g}: R(t={args.tune_target:g}) = {finals[-1]:.4f},"
                  f" R(T) = {risk[-1]:.4f}  [{time.time() - t0:.0f}s]", flush=True)
        chosen[method] = float(grid[int(np.argmin(finals))])
    print(f"chosen eta: {chosen}")
    np.savez_compressed(OUT / "tune.npz", **results)
    (OUT / "eta.json").write_text(json.dumps(chosen))


def chosen_eta(args):
    path = OUT / "eta.json"
    eta = json.loads(path.read_text()) if path.exists() else {}
    return (args.eta_adam or eta.get("adam", 0.1)), (args.eta_sgd or eta.get("sgd", 1.0))


def record_grid(t_min, t_max, dimension, per_decade=12):
    """Log-spaced record times rounded to integer steps (unique, starting at 0)."""
    steps = np.unique(np.round(log_grid(t_min, t_max, per_decade) * dimension).astype(int))
    return steps, steps / dimension


def deterministic_start(classes, means, scale=0.1, seed=0):
    """``Theta`` with ``Theta^T Theta = scale^2 Id`` and ``Theta^T Xbar = 0``: the
    ``D -> infinity`` limit of ``(B, O)`` at ``Theta_0 = scale N(0, 1/D)``. One ODE
    started here serves every ``D``; the finite-D fluctuation of ``Theta_0`` is
    then part of the measured finite-D error."""
    Z = jax.random.normal(jax.random.PRNGKey(seed + 99), (means.shape[0], classes))
    Z = Z - means @ jnp.linalg.solve(means.T @ means, means.T @ Z)
    return scale * jnp.linalg.qr(Z)[0]


def ode_on_grid(path, priors, means, times, *, cache, **kw):
    """Collapsed ODE from the deterministic start, per-class metrics, cached in ``path``."""
    if path.exists() and not kw.pop("overwrite", False):
        return dict(np.load(path))
    kw.pop("overwrite", None)
    t0 = time.time()
    out = solve(priors, deterministic_start(len(priors), means), means, times, **kw)
    loss, err = class_metrics(out["B"], out["O"])
    res = {"time": np.asarray(times), "loss": loss, "err": err,
           "Bdiag": np.asarray(jnp.diagonal(out["B"], axis1=-2, axis2=-1)),
           "Odiag": np.asarray(jnp.diagonal(out["O"], axis1=-2, axis2=-1)),
           "coef": np.asarray(out["coef_diag"]), "steps": np.asarray(out["steps"]),
           "seconds": np.array(time.time() - t0)}
    if cache:
        np.savez_compressed(path, **res)
    return res


def interp_log(t_new, t, y):
    """Interpolate ``y(t)`` (first axis) in ``log t``; ``t[0] = 0`` handled linearly."""
    t_new, t, y = np.asarray(t_new), np.asarray(t), np.asarray(y)
    x = np.log(np.maximum(t, 1e-12))
    xn = np.log(np.maximum(t_new, 1e-12))
    flat = y.reshape(len(t), -1)
    out = np.stack([np.interp(xn, x, flat[:, j]) for j in range(flat.shape[1])], axis=-1)
    return out.reshape((len(t_new),) + y.shape[1:])


def validation_horizon(classes, args):
    return {8: args.horizon_small, 32: args.horizon_mid}.get(classes, args.horizon_large)


def run_validate(args):
    eta, _ = chosen_eta(args)
    cases = [(C, a, b, args.val_dimension) for C in args.val_classes
             for a in args.val_alphas for b in args.val_batches]
    cases += [(C, a, b, D) for (C, a, b) in args.dsweep for D in args.val_dimensions
              if (C, a, b, D) not in cases]
    for classes, alpha, batch, dimension in cases:
        name = f"validate_C{classes}_a{alpha:g}_b{batch}_D{dimension}"
        if (OUT / f"{name}.npz").exists() and not args.overwrite:
            continue
        horizon = validation_horizon(classes, args)
        priors, data, theta0 = zipf_model(classes, alpha, dimension)
        ode = ode_on_grid(OUT / f"validate_ode_C{classes}_a{alpha:g}_b{batch}.npz", priors,
                          data.means, log_grid(args.t_min, horizon), cache=True, lr=eta,
                          beta1=args.beta1, beta2=args.beta2, eps=args.eps, batch=batch)
        steps, times = record_grid(args.t_min, horizon, dimension)
        t0 = time.time()
        B, O = simulate_runs(priors, data, theta0, steps, lr=eta, beta1=args.beta1,
                             beta2=args.beta2, eps=args.eps, batch=batch, runs=args.runs,
                             seed=classes + batch)
        sim_loss, sim_err = class_metrics(B, O)
        seconds = np.array([float(ode["seconds"]), time.time() - t0])
        out = {"time": times, "priors": np.asarray(priors), "eta": eta,
               "ode_loss": interp_log(times, ode["time"], ode["loss"]),
               "ode_err": interp_log(times, ode["time"], ode["err"]),
               "ode_Bdiag": interp_log(times, ode["time"], ode["Bdiag"]),
               "sim_loss": sim_loss, "sim_err": sim_err,
               "sim_Bdiag": np.asarray(jnp.diagonal(B, axis1=-2, axis2=-1)),
               "sim_Odiag": np.asarray(jnp.diagonal(O, axis1=-2, axis2=-1)),
               "seconds": seconds}
        np.savez_compressed(OUT / f"{name}.npz", **out)
        pri = out["priors"]
        gap = np.abs(total(sim_loss, pri).mean(0) - total(out["ode_loss"], pri))
        se = total(sim_loss, pri).std(0) / np.sqrt(args.runs)
        k = int(np.argmax(gap))
        print(f"{name}: sup mean gap {gap[k]:.4f} (se {se[k]:.4f}) at t={times[k]:.3g};"
              f" ODE {seconds[0]:.0f}s ({int(ode['steps'].sum())} steps), sim {seconds[1]:.0f}s",
              flush=True)


# --------------------------------------------------------------------------- #
# E1 (rates, timescales)
# --------------------------------------------------------------------------- #
def rate_configs(args):
    configs = [(b2, 1, args.eps) for b2 in (0.95, 0.99, 0.999)]
    configs += [(args.beta2, b, args.eps) for b in (1, 4, 16, 64)]
    configs += [(args.beta2, b, e) for b in (1, 16) for e in (1e-8, 1e-4, 1e-2)]
    return list(dict.fromkeys(configs))


def run_rates(args):
    """Frozen per-class rates of ``O_cc`` at ``Theta_0`` and a mid-training state."""
    eta, _ = chosen_eta(args)
    C = args.rates_classes
    priors, data, theta0 = zipf_model(C, 2.0, 4000)
    mid = solve(priors, theta0, data.means, log_grid(0.01, args.rates_mid_time), lr=eta,
                beta1=args.beta1, beta2=args.beta2, eps=args.eps, batch=1)
    states = {"init": (theta0.T @ theta0, theta0.T @ data.means),
              "mid": (mid["B"][-1], mid["O"][-1])}
    results = {"priors": np.asarray(priors), "mid_time": args.rates_mid_time}
    for name, (B, O) in states.items():
        results[f"{name}_B"], results[f"{name}_O"] = np.asarray(B), np.asarray(O)
        loss, err = class_metrics(B[None], O[None])
        results[f"{name}_loss"] = loss[0]
    for beta2, batch, eps in rate_configs(args):
        t0 = time.time()
        tag = f"b2{beta2:g}_b{batch}_e{eps:g}"
        layout, plan = make_plan(priors, args.beta1, beta2, batch, min_chunks=16)
        draws = moment_draws(C)
        for name, (B, O) in states.items():
            est, se = discounts.softmax_standard_errors(B, O, priors, plan, args.beta1, beta2,
                                                        eps, layout=layout)
            co = discounts.softmax_coefficients(B, O, priors, plan, args.beta1, beta2, eps,
                                                layout=layout)
            results[f"{name}_{tag}_adam"] = frozen_rate(co, O)
            results[f"{name}_{tag}_chi_relse"] = np.asarray(se[0] / jnp.abs(est[0]))
            for mode in ("psgd", "sgd"):
                cm = discounts.softmax_moment_coefficients(B, O, priors, draws, eps, batch,
                                                           mode=mode)
                results[f"{name}_{tag}_{mode}"] = frozen_rate(cm, O)
        results[f"events_{tag}"] = np.asarray(plan.counts).sum(0)
        print(f"rates {tag}: [{time.time() - t0:.0f}s]  max rel s.e. of chi_cc at init "
              f"{results[f'init_{tag}_chi_relse'].max():.3f}", flush=True)
    if args.rates_check:
        results.update(frozen_simulation_check(priors, data, theta0, args))
    np.savez_compressed(OUT / "rates.npz", **results)


def frozen_simulation_check(priors, data, theta0, args):
    """Direct frozen-parameter Adam drift of ``O_cc`` for a few classes at ``Theta_0``."""
    from jax import lax
    problem = problems.make_gmm_softmax(np.asarray(priors))
    classes = np.asarray(args.rates_check_classes)
    beta1, beta2, eps, batch = args.beta1, args.beta2, args.eps, 1
    stability = eps * jnp.ones((theta0.shape[0], 1))

    def chain(key):
        def step(carry, k):
            m, v, acc = carry
            g = simulate._mixture_grad(problem, data, theta0, k, batch)
            m = beta1 * m + (1 - beta1) * g
            v = beta2 * v + (1 - beta2) * g ** 2
            u = m / jnp.sqrt(v + stability)
            return (m, v, acc + jnp.sum(u[:, classes] * data.means[:, classes], axis=0)), None

        zero = jnp.zeros_like(theta0)
        burn = int(np.ceil(np.log(1e-8) / np.log(beta2)))
        (m, v, _), _ = lax.scan(step, (zero, zero, jnp.zeros(len(classes))),
                                jax.random.split(key, burn))
        (_, _, acc), _ = lax.scan(step, (m, v, jnp.zeros(len(classes))),
                                  jax.random.split(jax.random.fold_in(key, 1),
                                                   args.rates_check_steps))
        return -acc / args.rates_check_steps

    t0 = time.time()
    keys = jax.random.split(jax.random.PRNGKey(77), args.rates_check_chains)
    per_chain = np.asarray(jax.jit(jax.vmap(chain))(keys))
    print(f"frozen simulation check: classes {classes}, rate {per_chain.mean(0)} +- "
          f"{per_chain.std(0) / np.sqrt(len(per_chain))}  [{time.time() - t0:.0f}s]", flush=True)
    return {"check_classes": classes, "check_rate": per_chain.mean(0),
            "check_se": per_chain.std(0) / np.sqrt(len(per_chain))}


def timescale_configs(args):
    return [dict(beta2=args.beta2, batch=1, eps=args.eps), dict(beta2=0.95, batch=1, eps=args.eps),
            dict(beta2=args.beta2, batch=16, eps=args.eps), dict(beta2=args.beta2, batch=1, eps=1e-4)]


def run_timescales(args):
    """Per-class learning times from ODE trajectories (Adam, PSGD, SGD)."""
    eta_adam, eta_sgd = chosen_eta(args)
    C = args.timescale_classes
    priors, data, theta0 = zipf_model(C, 2.0, 4000)
    times = log_grid(0.01, args.timescale_horizon)
    results = {"priors": np.asarray(priors), "time": times}
    for cfg in timescale_configs(args):
        tag = f"b2{cfg['beta2']:g}_b{cfg['batch']}_e{cfg['eps']:g}"
        for mode, eta in (("adam", eta_adam), ("psgd", eta_adam), ("sgd", eta_sgd)):
            if mode == "sgd" and f"sgd_b{cfg['batch']}_loss" in results:
                continue
            t0 = time.time()
            res = solve(priors, theta0, data.means, times, lr=eta, beta1=args.beta1,
                        beta2=cfg["beta2"], eps=cfg["eps"], batch=cfg["batch"], mode=mode)
            loss, err = class_metrics(res["B"], res["O"])
            key = f"sgd_b{cfg['batch']}" if mode == "sgd" else f"{mode}_{tag}"
            results[f"{key}_loss"], results[f"{key}_err"] = loss, err
            results[f"{key}_Bdiag"] = np.asarray(jnp.diagonal(res["B"], axis1=-2, axis2=-1))
            results[f"{key}_coef"] = np.asarray(res["coef_diag"])
            t_loss, t_err = learning_times(times, loss, err)
            results[f"{key}_tc"], results[f"{key}_tc_err"] = t_loss, t_err
            print(f"timescales {key}: {np.isfinite(t_loss).sum()}/{C} classes learned by "
                  f"t={times[-1]:g}  [{time.time() - t0:.0f}s, "
                  f"{int(np.asarray(res['steps']).sum())} steps]", flush=True)
    if args.timescale_check:
        steps, sim_times = record_grid(0.01, args.timescale_check_horizon, 4000)
        B, O = simulate_runs(priors, data, theta0, steps, lr=eta_adam, beta1=args.beta1,
                             beta2=args.beta2, eps=args.eps, batch=1, runs=args.runs // 2)
        loss, err = class_metrics(B, O)
        results["check_time"], results["check_loss"], results["check_err"] = sim_times, loss, err
        tc = [learning_times(sim_times, l, e)[0] for l, e in zip(loss, err)]
        results["check_tc"] = np.array(tc)
    np.savez_compressed(OUT / "timescales.npz", **results)


# --------------------------------------------------------------------------- #
# E2 (exponents), E3 (batch), E4 (noise): ODE only, with spot-check simulations
# --------------------------------------------------------------------------- #
def local_exponent(times, values, smooth=3):
    """``-d log R / d log t`` on a log grid (centred differences, light smoothing)."""
    t, v = np.asarray(times[1:]), np.asarray(values[1:])
    slope = -np.gradient(np.log(v), np.log(t))
    if smooth > 1:
        kernel = np.ones(smooth) / smooth
        slope = np.convolve(np.pad(slope, smooth // 2, mode="edge"), kernel, mode="valid")
    return np.concatenate([[np.nan], slope])


def frontier(times, t_c):
    """Number of classes learned by each time (``t_c <= t``)."""
    t_c = np.where(np.isfinite(t_c), t_c, np.inf)
    return np.array([(t_c <= t).sum() for t in times])


def exponent_configs(args):
    base = dict(beta2=args.beta2, batch=1, eps=args.eps)
    configs = [base, dict(base, beta2=0.95), dict(base, beta2=0.999), dict(base, batch=16),
               dict(base, eps=1e-4)]
    return configs


def run_exponents(args):
    eta_adam, eta_sgd = chosen_eta(args)
    times = log_grid(0.01, args.exponent_horizon, per_decade=16)
    for classes in args.exponent_classes:
        for alpha in args.exponent_alphas:
            priors, data, _ = zipf_model(classes, alpha, 4000)
            for cfg in exponent_configs(args):
                if alpha != 2.0 and cfg != exponent_configs(args)[0]:
                    continue
                tag = (f"C{classes}_a{alpha:g}_b2{cfg['beta2']:g}_b{cfg['batch']}"
                       f"_e{cfg['eps']:g}")
                for mode, eta in (("adam", eta_adam), ("psgd", eta_adam), ("sgd", eta_sgd)):
                    if mode == "sgd" and cfg != exponent_configs(args)[0]:
                        continue
                    path = OUT / f"exponents_{mode}_{tag}.npz"
                    plan = None
                    if mode == "adam":
                        plan = make_plan(priors, args.beta1, cfg["beta2"], cfg["batch"],
                                         min_events=args.exponent_min_events)
                    res = ode_on_grid(path, priors, data.means, times, cache=True,
                                      overwrite=args.overwrite, lr=eta, beta1=args.beta1,
                                      beta2=cfg["beta2"], eps=cfg["eps"], batch=cfg["batch"],
                                      mode=mode, plan=plan)
                    print(f"exponents {mode} {tag}: R(T)={total(res['loss'][-1], priors):.4f}"
                          f"  [{float(res['seconds']):.0f}s, {int(res['steps'].sum())} steps]",
                          flush=True)


def run_batch(args):
    """Samples to reach target losses against b: fixed eta, sqrt(b) scaling, tuned eta."""
    eta_adam, _ = chosen_eta(args)
    classes, alpha = args.batch_classes, 2.0
    priors, data, _ = zipf_model(classes, alpha, 4000)
    times = log_grid(0.01, args.batch_horizon, per_decade=16)
    for eps in args.batch_eps:
        for b in args.batch_sizes:
            for rule, eta in (("fixed", eta_adam), ("sqrt", eta_adam * np.sqrt(b))):
                path = OUT / f"batch_C{classes}_e{eps:g}_b{b}_{rule}.npz"
                res = ode_on_grid(path, priors, data.means, times, cache=True,
                                  overwrite=args.overwrite, lr=eta, beta1=args.beta1,
                                  beta2=args.beta2, eps=eps, batch=b)
                print(f"batch eps={eps:g} b={b} {rule} eta={eta:g}:"
                      f" R(T)={total(res['loss'][-1], priors):.4f} [{float(res['seconds']):.0f}s]",
                      flush=True)


def run_noise(args):
    """Rare-class plateaus of B_cc and ell_c against eta (Adam and SGD)."""
    classes, alpha = args.noise_classes, 2.0
    priors, data, _ = zipf_model(classes, alpha, 4000)
    times = log_grid(0.01, args.noise_horizon, per_decade=12)
    for mode, grid in (("adam", args.noise_adam_etas), ("sgd", args.noise_sgd_etas)):
        for eta in grid:
            path = OUT / f"noise_{mode}_C{classes}_eta{eta:g}.npz"
            res = ode_on_grid(path, priors, data.means, times, cache=True,
                              overwrite=args.overwrite, lr=eta, beta1=args.beta1,
                              beta2=args.beta2, eps=args.eps, batch=1, mode=mode)
            print(f"noise {mode} eta={eta:g}: R(T)={total(res['loss'][-1], priors):.4f},"
                  f" max B_cc(T)={res['Bdiag'][-1].max():.3g} [{float(res['seconds']):.0f}s]",
                  flush=True)


# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #
def save(fig, name):
    path = OUT / name
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight", metadata={"CreationDate": None})
    fig.savefig(path.with_suffix(".png"), dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"saved -> {path}.{{pdf,png}}")


def band(ax, x, runs, color, label=None, marker="o"):
    mean = runs.mean(0)
    lo, hi = np.percentile(runs, [10, 90], axis=0)
    ax.errorbar(x, mean, yerr=[np.maximum(mean - lo, 0), np.maximum(hi - mean, 0)], fmt=marker,
                color=color, ms=4, capsize=2, lw=1, label=label)


def plot_validate(args):
    files = sorted(OUT.glob("validate_*.npz"))
    if not files:
        return
    cases = {f.stem: dict(np.load(f)) for f in files}
    main = {k: v for k, v in cases.items() if k.endswith(f"_D{args.val_dimension}")}
    fig, axes = plt.subplots(1, 4, figsize=(24, 5.6), constrained_layout=True)
    markers = {1: "o", 8: "s", 32: "^"}
    colors = {8: "#1f77b4", 32: "#ff7f0e", 128: "#2ca02c"}
    for name, r in main.items():
        C, b = int(name.split("_C")[1].split("_")[0]), int(name.split("_b")[1].split("_")[0])
        pri = r["priors"]
        for ax, key in ((axes[0], "loss"), (axes[1], "err")):
            sim = total(r[f"sim_{key}"], pri)
            ode = total(r[f"ode_{key}"], pri)
            ax.plot(ode[1:], sim.mean(0)[1:], markers.get(b, "o"), color=colors.get(C, "k"),
                    ms=4, alpha=0.7)
    for ax, lab in ((axes[0], "cross-entropy"), (axes[1], "0-1 error")):
        lims = ax.get_xlim()
        ax.plot(lims, lims, "k--", lw=1)
        ax.set_xlabel(f"ODE {lab}")
        ax.set_ylabel(f"Adam {lab} (mean of {args.runs} runs)")
        ax.set_xscale("log")
        ax.set_yscale("log")
    for C, c in colors.items():
        axes[0].plot([], [], "o", color=c, label=f"$C={C}$")
    for b, m in markers.items():
        axes[0].plot([], [], m, color="0.4", label=f"$b={b}$")
    axes[0].legend(frameon=False, fontsize=11)
    axes[0].set_title(f"Risk vs predicted, $D={args.val_dimension}$")
    axes[1].set_title("0-1 error vs predicted")

    key = f"validate_C128_a2_b1_D{args.val_dimension}"
    if key in cases:
        r = cases[key]
        ax = axes[2]
        show = [0, 3, 15, 63, 127]
        cols = plt.cm.viridis(np.linspace(0.05, 0.9, len(show)))
        for c, col in zip(show, cols):
            ax.plot(r["time"][1:], r["ode_loss"][1:, c], color=col, lw=2,
                    label=fr"$c={c + 1}$, $\pi_c$={r['priors'][c]:.1e}")
            band(ax, r["time"][1::2], r["sim_loss"][:, 1::2, c], col, marker=".")
        ax.set_xscale("log")
        ax.set_xlabel("$t$")
        ax.set_ylabel(r"$\ell_c(t)$")
        ax.set_title(r"Per-class loss, $C=128$, $\alpha=2$, $b=1$")
        ax.legend(frameon=False, fontsize=10)

    ax = axes[3]
    for (C, a, b), col in zip(args.dsweep, ("#1f77b4", "#d62728", "#2ca02c")):
        dims, med, mean_gap = [], [], []
        for D in args.val_dimensions:
            name = f"validate_C{C}_a{a:g}_b{b}_D{D}"
            if name not in cases:
                continue
            r = cases[name]
            pri = r["priors"]
            ode = np.interp(r["time"], r["time"], total(r["ode_loss"], pri))
            sim = total(r["sim_loss"], pri)
            dims.append(D)
            med.append(np.median(np.abs(sim - ode[None]).max(1)))
            mean_gap.append(np.abs(sim.mean(0) - ode).max())
        if dims:
            ax.plot(dims, med, "o-", color=col, label=f"per run, C={C}, b={b}")
            ax.plot(dims, mean_gap, "s--", color=col, label=f"mean, C={C}, b={b}")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("$D$")
    ax.set_ylabel(r"$\sup_t|\mathcal{R}-\mathscr{R}|$")
    ax.set_title("Gap vs dimension")
    ax.legend(frameon=False, fontsize=9)
    for ax in axes:
        polish_axis(ax)
    save(fig, "validate")


def plot_rates(args):
    path = OUT / "rates.npz"
    if not path.exists():
        return
    r = dict(np.load(path))
    pi = r["priors"]
    groups = [("beta2", [(b2, 1, args.eps) for b2 in (0.95, 0.99, 0.999)]),
              ("batch", [(args.beta2, b, args.eps) for b in (1, 4, 16, 64)]),
              ("eps", [(args.beta2, 16, e) for e in (1e-8, 1e-4, 1e-2)])]
    for state in ("init", "mid"):
        fig, axes = plt.subplots(1, 3, figsize=(21, 6), constrained_layout=True)
        for ax, (name, cfgs) in zip(axes, groups):
            cols = plt.cm.plasma(np.linspace(0.05, 0.8, len(cfgs)))
            for (b2, b, e), col in zip(cfgs, cols):
                tag = f"b2{b2:g}_b{b}_e{e:g}"
                if f"{state}_{tag}_adam" not in r:
                    continue
                label = {"beta2": fr"$\beta_2={b2:g}$", "batch": f"$b={b}$",
                         "eps": fr"$\epsilon={e:g}$"}[name]
                ax.plot(pi, np.abs(r[f"{state}_{tag}_adam"]), "o", color=col, ms=4, label=label)
                ax.plot(pi, np.abs(r[f"{state}_{tag}_psgd"]), "-", color=col, lw=1.4, alpha=0.8)
                pred = predicted_rates(pi, args.beta1, b2, e, b)
                if state == "init":
                    ax.plot(pi, pred["rare"], ":", color=col, lw=1.4)
                    for kink, ls in ((pred["kink_beta2"], "--"), (pred["kink_eps"], "-.")):
                        if pi.min() < kink < pi.max():
                            ax.axvline(kink, color=col, ls=ls, lw=1, alpha=0.7)
            sgd_tag = f"b2{cfgs[0][0]:g}_b{cfgs[0][1]}_e{cfgs[0][2]:g}"
            ax.plot(pi, np.abs(r[f"{state}_{sgd_tag}_sgd"]), "k-", lw=2, alpha=0.5, label="SGD")
            x = np.array([pi.min(), pi.max()])
            ax.plot(x, 3 * np.sqrt(x / x[1]) * np.abs(r[f"{state}_{sgd_tag}_adam"][0]), "k--",
                    lw=1, alpha=0.4, label="slope 1/2")
            ax.plot(x, x / x[1] * np.abs(r[f"{state}_{sgd_tag}_adam"][0]), "k:", lw=1, alpha=0.4,
                    label="slope 1")
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlabel(r"$\pi_c$")
            ax.set_ylabel(r"$|dO_{cc}/dt| / \eta$")
            ax.legend(frameon=False, fontsize=10)
            polish_axis(ax)
        axes[0].set_title(r"Sweep $\beta_2$ ($b=1$)")
        axes[1].set_title(r"Sweep $b$ ($\beta_2=%g$)" % args.beta2)
        axes[2].set_title(r"Sweep $\epsilon$ ($b=16$)")
        where = r"$\Theta_0$" if state == "init" else f"$t={float(r['mid_time']):g}$"
        fig.suptitle(f"Frozen per-class rates at {where}: markers Adam, lines PSGD,"
                     " dotted the rare-event law")
        save(fig, f"rates_{state}")


def plot_timescales(args):
    path = OUT / "timescales.npz"
    if not path.exists():
        return
    r = dict(np.load(path))
    pi, times = r["priors"], r["time"]
    cfgs = timescale_configs(args)
    fig, axes = plt.subplots(1, 4, figsize=(26, 6), constrained_layout=True)
    ax = axes[0]
    cols = plt.cm.plasma(np.linspace(0.05, 0.8, len(cfgs)))
    for cfg, col in zip(cfgs, cols):
        tag = f"b2{cfg['beta2']:g}_b{cfg['batch']}_e{cfg['eps']:g}"
        ax.plot(pi, r[f"adam_{tag}_tc"], "o", color=col, ms=4,
                label=fr"Adam $\beta_2={cfg['beta2']:g}$, $b={cfg['batch']}$, $\epsilon={cfg['eps']:g}$")
        ax.plot(pi, r[f"psgd_{tag}_tc"], "-", color=col, lw=1.2, alpha=0.8)
        pred = predicted_rates(pi, args.beta1, cfg["beta2"], cfg["eps"], cfg["batch"])
        ax.axvline(pred["kink_beta2"], color=col, ls="--", lw=1, alpha=0.6)
    for b in sorted({c["batch"] for c in cfgs}):
        ax.plot(pi, r[f"sgd_b{b}_tc"], "k-", lw=2, alpha=0.5, label=f"SGD $b={b}$")
    if "check_tc" in r:
        tc = r["check_tc"]
        ax.plot(pi, np.nanmedian(tc, 0), "x", color="k", ms=6, label="Adam runs (median)")
    x = np.array([pi.min(), pi.max()])
    ref = np.nanmedian(r[f"adam_b2{cfgs[0]['beta2']:g}_b1_e{cfgs[0]['eps']:g}_tc"][:3])
    ax.plot(x, ref * (x / pi[1]) ** -0.5, "k--", lw=1, alpha=0.5, label="slope -1/2")
    ax.plot(x, ref * (x / pi[1]) ** -1.0, "k:", lw=1, alpha=0.5, label="slope -1")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$\pi_c$")
    ax.set_ylabel(r"$t_c$ ($\ell_c \leq \ell_c(0)/2$)")
    ax.set_title("Per-class learning times")
    ax.legend(frameon=False, fontsize=9)
    tag = f"b2{cfgs[0]['beta2']:g}_b{cfgs[0]['batch']}_e{cfgs[0]['eps']:g}"
    b, b2 = cfgs[0]["batch"], cfgs[0]["beta2"]
    scalings = [("adam", np.sqrt(b * pi), r"$t\sqrt{b\pi_c}$"),
                ("adam", b * pi / np.sqrt(1 - b2), r"$t\,b\pi_c/\sqrt{1-\beta_2}$"),
                ("sgd", pi, r"$t\pi_c$")]
    for ax, (method, scale, lab) in zip(axes[1:], scalings):
        loss = r[f"adam_{tag}_loss"] if method == "adam" else r[f"sgd_b{b}_loss"]
        colors = plt.cm.viridis(np.linspace(0, 1, len(pi)))
        for c in range(len(pi)):
            ax.plot(times[1:] * scale[c], loss[1:, c] / loss[0, c], color=colors[c], lw=1)
        ax.set_xscale("log")
        ax.set_xlabel(lab)
        ax.set_ylabel(r"$\ell_c/\ell_c(0)$")
        ax.set_title(f"{'Adam' if method == 'adam' else 'SGD'} collapse")
    for ax in axes:
        polish_axis(ax)
    save(fig, "timescales")


def run_plot(args):
    plot_validate(args)
    plot_rates(args)
    plot_timescales(args)


# --------------------------------------------------------------------------- #
def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=["selftest", "timings", "precision", "tune", "validate", "rates",
                                       "timescales", "exponents", "batch", "noise", "plot"])
    p.add_argument("--timing-classes", type=int, nargs="+", default=[8, 32, 128, 256])
    p.add_argument("--timing-beta2", type=float, nargs="+", default=[0.95, 0.99, 0.999])
    p.add_argument("--timing-batches", type=int, nargs="+", default=[1, 16, 64])
    p.add_argument("--timing-horizon", type=float, default=1000.0)
    p.add_argument("--precision-classes", type=int, nargs="+", default=[32, 128])
    # shared hyper-parameters
    p.add_argument("--beta1", type=float, default=0.9)
    p.add_argument("--beta2", type=float, default=0.99)
    p.add_argument("--eps", type=float, default=1e-8)
    p.add_argument("--eta-adam", type=float, default=None)
    p.add_argument("--eta-sgd", type=float, default=None)
    p.add_argument("--runs", type=int, default=16)
    p.add_argument("--overwrite", action="store_true")
    # tuning
    p.add_argument("--tune-classes", type=int, default=32)
    p.add_argument("--tune-alpha", type=float, default=1.5)
    p.add_argument("--tune-horizon", type=float, default=1000.0)
    p.add_argument("--tune-target", type=float, default=100.0)
    p.add_argument("--adam-etas", type=float, nargs="+", default=[0.03, 0.1, 0.3, 1.0])
    p.add_argument("--sgd-etas", type=float, nargs="+", default=[0.3, 1.0, 3.0, 10.0])
    # E0
    p.add_argument("--val-classes", type=int, nargs="+", default=[8, 32, 128])
    p.add_argument("--val-alphas", type=float, nargs="+", default=[1.2, 1.5, 2.0])
    p.add_argument("--val-batches", type=int, nargs="+", default=[1, 8, 32])
    p.add_argument("--val-dimension", type=int, default=4000)
    p.add_argument("--val-dimensions", type=int, nargs="+", default=[2000, 4000, 8000])
    p.add_argument("--dsweep", type=json.loads,
                   default=[[32, 2.0, 1], [32, 2.0, 32], [128, 2.0, 8]])
    p.add_argument("--t-min", type=float, default=0.01)
    p.add_argument("--horizon-small", type=float, default=30.0)
    p.add_argument("--horizon-mid", type=float, default=300.0)
    p.add_argument("--horizon-large", type=float, default=1000.0)
    # E1
    p.add_argument("--rates-classes", type=int, default=128)
    p.add_argument("--rates-mid-time", type=float, default=10.0)
    p.add_argument("--rates-check", action="store_true")
    p.add_argument("--rates-check-classes", type=int, nargs="+", default=[2, 7, 24])
    p.add_argument("--rates-check-steps", type=int, default=50000)
    p.add_argument("--rates-check-chains", type=int, default=256)
    p.add_argument("--timescale-classes", type=int, default=64)
    p.add_argument("--timescale-horizon", type=float, default=3000.0)
    p.add_argument("--timescale-check", action="store_true")
    p.add_argument("--timescale-check-horizon", type=float, default=300.0)
    # E2-E4
    p.add_argument("--exponent-classes", type=int, nargs="+", default=[128])
    p.add_argument("--exponent-alphas", type=float, nargs="+", default=[1.25, 1.5, 2.0, 3.0])
    p.add_argument("--exponent-horizon", type=float, default=1e4)
    p.add_argument("--exponent-min-events", type=int, default=1000)
    p.add_argument("--batch-classes", type=int, default=64)
    p.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 4, 16, 64])
    p.add_argument("--batch-eps", type=float, nargs="+", default=[1e-8, 1e-4])
    p.add_argument("--batch-horizon", type=float, default=3000.0)
    p.add_argument("--noise-classes", type=int, default=64)
    p.add_argument("--noise-adam-etas", type=float, nargs="+", default=[0.03, 0.1, 0.3, 1.0])
    p.add_argument("--noise-sgd-etas", type=float, nargs="+", default=[0.3, 1.0, 3.0, 10.0])
    p.add_argument("--noise-horizon", type=float, default=3000.0)
    p.add_argument("--output-dir", type=Path, default=OUT)
    return p.parse_args()


def main():
    global OUT
    args = parse_args()
    OUT = args.output_dir
    OUT.mkdir(parents=True, exist_ok=True)
    {"selftest": run_selftest, "timings": run_timings, "precision": run_precision,
     "tune": run_tune, "validate": run_validate, "rates": run_rates,
     "timescales": run_timescales, "exponents": run_exponents, "batch": run_batch,
     "noise": run_noise, "plot": run_plot}[args.command](args)


if __name__ == "__main__":
    main()
