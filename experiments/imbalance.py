#!/usr/bin/env python3
"""Series I: class imbalance, Adam vs SGD -- per-class and risk convergence rates.

Models (``experiments/imbalance/README.md``): ``C`` classes, priors ``pi_c ~ c^{-alpha}``,
means ``r q_c`` (``r = 5``, ``q_c`` orthonormal), noise ``K = kappa Id``, softmax
cross-entropy with ``C`` logits.

* ``m0``: no bias. Collapsed isotropic ODE (Adam: psi-route; SGD: path-free moments)
  plus direct simulations.
* ``m1``: ``m0`` plus a learned bias trained by the same optimizer and ``lr``.
  Direct simulation only (the Adam bias drift has no ODE estimator yet).

``K = kappa Id`` is exactly ``K = Id`` with ``r' = r / sqrt(kappa)`` and
``eta' = sqrt(kappa) eta`` (Adam; its K-scaled eps is invariant) or ``kappa eta``
(SGD), with ``B``, ``O`` unchanged (rescale ``x' = x / sqrt(kappa)``,
``Theta' = sqrt(kappa) Theta``). The ODE runs in the primed units; the
simulator runs ``K = kappa Id`` directly (the ``selftest`` checks the identity).

Subcommands: ``selftest | ode | sim | frozen | frozen-bias | probe | analyze | plot``.
Outputs: ``figs/experiments/imbalance/{ode,sim,frozen,analysis,logs}``.
"""
from __future__ import annotations

import argparse
import json
import os
import runpy
import subprocess
import sys
import time
import zlib
from pathlib import Path
from typing import NamedTuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figs" / "experiments" / "imbalance"
ANALYSIS = Path(__file__).resolve().parent / "imbalance" / "analysis"

R = 5.0
BETA1 = 0.9
EPS = 1e-8
BATCH = 1
ETA = {"adam": 0.3, "sgd": 3.0}
SAFETY = 1.0
B_START = 0.01          # Theta_0^T Theta_0 = 0.01 Id (deterministic start, original units)
PER_DECADE = 16
T_MIN = 0.01

CLASSES = (8, 32, 128)
ALPHAS = (1.0, 2.0)
BETA2S = (0.9, 0.95, 0.99, 0.999)
KAPPAS = (1.0, 0.1)
MODELS = ("m0",)          # m1 (learned bias, simulation only) is still available via --models
OPTS = ("adam", "sgd")
ETA_FILE = OUT / "eta_tuned.json"


def eta_of(opt, kappa, etaset="base"):
    """``eta`` (original units) of a learning-rate set: ``base`` = E-series values,
    ``tuned`` = per-``kappa`` values from ``imbalance.py tune`` (``eta_tuned.json``)."""
    if etaset == "base":
        return ETA[opt]
    table = json.loads(ETA_FILE.read_text())
    return float(table[f"{kappa:g}"][opt])


# --------------------------------------------------------------------------- #
# Runs, names and grids (numpy only)
# --------------------------------------------------------------------------- #
class Run(NamedTuple):
    model: str
    opt: str
    kappa: float
    classes: int
    alpha: float
    beta2: float
    horizon: float
    safety: float = SAFETY
    etaset: str = "base"

    @property
    def eta(self):
        return eta_of(self.opt, self.kappa, self.etaset)

    @property
    def t_min(self):
        """First record time: 0.01, lowered by whole decades when ``eta`` exceeds the base
        value, so that classes learned early (large ``eta``) are still resolved."""
        ratio = self.eta / ETA[self.opt]
        return T_MIN * 10.0 ** -max(0, int(np.ceil(np.log10(ratio) - 1e-9)))


def fmt_T(T):
    return f"{T:.6g}"


def stem(run: Run, dimension=None, kind="ode"):
    opt = (f"b1{BETA1:g}_b2{run.beta2:g}_b{BATCH}_e{EPS:g}" if run.opt == "adam"
           else f"b{BATCH}")
    name = (f"{run.model}_{run.opt}_k{run.kappa:g}_C{run.classes}_a{run.alpha:g}_{opt}"
            f"_eta{run.eta:g}" + (f"_tm{run.t_min:g}" if run.t_min != T_MIN else "")
            + f"_T{fmt_T(run.horizon)}")
    if kind == "ode":
        return name + f"_s{run.safety:g}"
    return name + f"_D{dimension}"


def grid(T, t_min=T_MIN, per_decade=PER_DECADE):
    """Nested log grid ``{0} U t_min 10^(k/per_decade)``; ``T`` must lie on it."""
    k = np.log10(T / t_min) * per_decade
    K = int(round(k))
    assert abs(k - K) < 1e-6, f"horizon {T} is not on the 1/{per_decade}-decade grid"
    return np.concatenate([[0.0], t_min * 10.0 ** (np.arange(K + 1) / per_decade)])


def ode_horizon(opt, classes):
    """Default horizons; ``C = 128`` Adam starts at ``10^2.5`` and is extended selectively."""
    if classes == 128:
        return 1e4 if opt == "sgd" else 10 ** 2.5
    return 10 ** 3.5 if (opt == "sgd" and classes == 32) else 1e3


def sim_horizon(model, opt, classes, alpha):
    if opt == "sgd" and classes == 128:
        return 10 ** 3.5
    return 10 ** 3.5 if (opt == "sgd" and classes == 32 and alpha == 2.0) else 1e3


def matrix(models=MODELS, kappas=KAPPAS, classes=CLASSES, alphas=ALPHAS, opts=OPTS,
           beta2s=BETA2S, kind="ode", etaset="base"):
    runs = []
    for model in models:
        for kappa in kappas:
            for C in classes:
                for alpha in alphas:
                    for opt in opts:
                        for beta2 in (beta2s if opt == "adam" else (0.0,)):
                            T = (ode_horizon(opt, C) if kind == "ode"
                                 else sim_horizon(model, opt, C, alpha))
                            runs.append(Run(model, opt, kappa, C, alpha, beta2, T,
                                            etaset=etaset))
    return runs


def primed(opt, kappa, eta):
    """``(r', eta')`` of the equivalent ``K = Id`` problem."""
    return R / np.sqrt(kappa), eta * (np.sqrt(kappa) if opt == "adam" else kappa)


def git_hash():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                                       text=True).strip()
    except Exception:
        return "?"


def log_run(record):
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "runs.jsonl", "a") as fh:
        fh.write(json.dumps({**record, "git": git_hash()}) + "\n")


# --------------------------------------------------------------------------- #
# GPU part (imports JAX lazily)
# --------------------------------------------------------------------------- #
def jax_modules():
    sys.path.insert(0, str(ROOT))
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/adam-matplotlib")
    from experiments import zipf_scaling as zs   # enables float64
    import jax
    import jax.numpy as jnp
    # persistent compilation cache: runs differing only in kappa/alpha reuse compiled code
    jax.config.update("jax_compilation_cache_dir", str(OUT / "jaxcache"))
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 5.0)
    import discounts
    import dynamics
    return zs, jax, jnp, discounts, dynamics


def adam_plan(priors, beta2, seed=0, legacy=False):
    """``zipf_scaling.make_plan`` (psi route, ``2^18`` batches, window 1024 / 8192), but for
    ``C > 32`` with all paths in one chunk: at ``C = 128`` the default 3-chunk layout takes
    ~500 s to compile (vs 89 s), for the same cost per call. ``C <= 32`` is unchanged.
    ``legacy=True`` forces the ``zipf_scaling.make_plan`` layout (same Monte Carlo draws as
    the E-series caches, e.g. for regression tests at ``C = 64``)."""
    zs, jax, jnp, discounts, dynamics = jax_modules()
    if legacy or len(priors) <= 32:
        return zs.make_plan(jnp.asarray(priors), BETA1, beta2, BATCH, seed=seed)
    return discounts.softmax_psi_plan(jax.random.PRNGKey(seed), jnp.asarray(priors), BETA1,
                                      beta2, batch=BATCH, plain_batches=max(2 ** 14, 2 ** 18 // BATCH),
                                      window=1024 if beta2 <= 0.99 else 8192,
                                      chunk_elements=2 ** 27)


def coefficients_for(opt, priors, beta2, seed=0, legacy=False):
    """``(fn, aux)`` for ``dynamics.run_mode_ode_logtime``."""
    zs, jax, jnp, discounts, dynamics = jax_modules()
    priors = jnp.asarray(priors)
    if opt == "adam":
        layout, arrays = adam_plan(priors, beta2, seed=seed, legacy=legacy)
        return zs.coefficient_fn(layout, "adam", False), (priors, arrays, BETA1, beta2, EPS)
    return zs.moment_fn("sgd"), (priors, zs.moment_draws(len(priors), seed=seed), EPS,
                                 float(BATCH))


def solve_state(priors, B0, O0, times, *, opt, beta2, kappa, safety, seed=0, eta=None,
                legacy_plan=False):
    """Collapsed ``K = Id`` ODE in primed units, started from ``(B0, O0)`` at ``times[0]``."""
    zs, jax, jnp, discounts, dynamics = jax_modules()
    C = len(priors)
    r, lr = primed(opt, kappa, ETA[opt] if eta is None else eta)
    d = 1000                       # cancels in the collapsed ODE (w_iso = w_cov = d)
    modes, _ = dynamics.isotropic_modes(jnp.zeros((d, C)), jnp.zeros((d, 0)),
                                        jnp.zeros((d, C)))
    initial = ((jnp.asarray(B0)[None], jnp.zeros((1, C, 0)), jnp.asarray(O0)[None]),
               (jnp.zeros((1, 0, 0)), jnp.zeros((1, 0, C)), (r ** 2 * jnp.eye(C))[None]))
    fn, aux = coefficients_for(opt, priors, beta2, seed, legacy=legacy_plan)
    return dynamics.run_mode_ode_logtime(fn, aux, modes, initial, lr, times, dimension=d,
                                         safety=safety)


def ode_path(run):
    return OUT / "ode" / f"{stem(run)}.npz"


def ode_cached(run: Run, *, extend=False, overwrite=False, full=True):
    """ODE trajectory of an ``m0`` run, cached; ``extend`` continues the longest
    cached shorter run with the same parameters (exact: the solver restarts from
    any state on the nested grid)."""
    zs, jax, jnp, discounts, dynamics = jax_modules()
    path = ode_path(run)
    if path.exists() and not overwrite:
        return dict(np.load(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    priors = zs.zipf_priors(run.classes, run.alpha)
    times = grid(run.horizon, t_min=run.t_min)
    C = run.classes
    start, prev = 0, None
    if extend:
        shorter = sorted(p for p in path.parent.glob(stem(run._replace(horizon=1)).replace(
            "_T1_", "_T*_") + ".npz") if p != path)
        best = None
        for p in shorter:
            d = dict(np.load(p))
            if d["time"][-1] < run.horizon and "B" in d and (
                    best is None or d["time"][-1] > best["time"][-1]):
                best = d
        if best is not None:
            prev = best
            start = len(best["time"]) - 1
            assert np.allclose(times[:start + 1], best["time"])
    t0 = time.time()
    if prev is None:
        B0, O0 = B_START * run.kappa * np.eye(C), np.zeros((C, C))
    else:
        B0, O0 = prev["B"][-1], prev["O"][-1]
    out = solve_state(priors, B0, O0, times[start:], opt=run.opt, beta2=run.beta2,
                      kappa=run.kappa, safety=run.safety, eta=run.eta)
    B, O = np.asarray(out["B"]), np.asarray(out["O"])
    loss, err = zs.class_metrics(jnp.asarray(B), jnp.asarray(O))
    res = {"time": times[start:], "loss": loss, "err": err, "B": B, "O": O,
           "coef": np.asarray(out["coef_diag"]), "steps": np.asarray(out["steps"])}
    seconds = time.time() - t0
    if prev is not None:
        res = {k: (np.concatenate([prev[k], v[1:]]) if k != "steps"
                   else np.concatenate([prev[k], v])) for k, v in res.items()}
        seconds += float(prev["seconds"])
    res["priors"] = priors
    res["seconds"] = np.array(seconds)
    np.savez_compressed(path, **res)
    log_run({"kind": "ode", "stem": stem(run), "seconds": seconds,
             "steps": int(res["steps"].sum()), "extended": prev is not None})
    print(f"ode {stem(run)}: {int(res['steps'].sum())} steps, {seconds:.0f}s"
          f"{' (extended)' if prev is not None else ''}", flush=True)
    return res


def sim_seed(run, dimension):
    return zlib.crc32(stem(run, dimension, "sim").encode()) % 100_000


def sim_path(run, dimension):
    return OUT / "sim" / f"{stem(run, dimension, 'sim')}.npz"


def sim_cached(run: Run, dimension, runs, *, overwrite=False):
    """Direct simulations (``K = kappa Id``, original units) recorded on the
    ``1/16``-decade grid; stores per-run class losses and order-parameter diagonals."""
    zs, jax, jnp, discounts, dynamics = jax_modules()
    path = sim_path(run, dimension)
    if path.exists() and not overwrite:
        return dict(np.load(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    priors, data, theta0 = zs.zipf_model(run.classes, run.alpha, dimension, noise=run.kappa)
    steps, times = zs.record_grid(T_MIN, run.horizon, dimension, per_decade=PER_DECADE)
    bias = run.model == "m1"
    t0 = time.time()
    out = zs.simulate_runs(priors, data, theta0, steps, lr=run.eta, beta1=BETA1,
                           beta2=run.beta2 if run.opt == "adam" else 0.99, eps=EPS,
                           batch=BATCH, runs=runs, sgd=run.opt == "sgd",
                           seed=sim_seed(run, dimension), bias=bias)
    jax.block_until_ready(out)
    sim_seconds = time.time() - t0
    B, O = out[0], out[1]
    beta = out[2] if bias else None
    loss, err = zs.class_metrics(B, O, bias=beta)
    res = {"time": times, "steps": steps, "loss": loss, "err": err,
           "Bdiag": np.asarray(jnp.diagonal(B, axis1=-2, axis2=-1)),
           "Odiag": np.asarray(jnp.diagonal(O, axis1=-2, axis2=-1)),
           "B_last": np.asarray(B[:, -1]), "O_last": np.asarray(O[:, -1]),
           "priors": np.asarray(priors), "seconds": np.array(sim_seconds),
           "dimension": np.array(dimension)}
    if bias:
        res["beta"] = np.asarray(beta)
    np.savez_compressed(path, **res)
    log_run({"kind": "sim", "stem": stem(run, dimension, "sim"), "seconds": sim_seconds,
             "steps": int(steps[-1]), "runs": runs})
    print(f"sim {stem(run, dimension, 'sim')}: {runs} runs x {steps[-1]} steps,"
          f" {sim_seconds:.0f}s", flush=True)
    return res


# --------------------------------------------------------------------------- #
# Frozen drifts
# --------------------------------------------------------------------------- #
def drift_matrix(co, O, r):
    """``dO/dt / eta = -(phi^T O + r^2 chi^T)`` (``K = Id``, orthonormal means)."""
    return -(np.asarray(co["phi"]).T @ np.asarray(O) + r ** 2 * np.asarray(co["chi"]).T)


def margin_rate(M):
    """Drift of the logit margin ``O_cc - mean_{k != c} O_kc`` from a drift matrix."""
    C = M.shape[0]
    return np.diag(M) - (M.sum(0) - np.diag(M)) / (C - 1)


def frozen_rates(priors, B, O, *, beta2, kappa, seeds, margin=False):
    """``|dO_cc/dt| / eta`` (original ``eta`` units) for Adam over plan seeds, and SGD;
    with ``margin=True`` also the margin drifts (Adam mean, s.e.; SGD)."""
    zs, jax, jnp, discounts, dynamics = jax_modules()
    priors = jnp.asarray(priors)
    r_a, lr_a = primed("adam", kappa, ETA["adam"])
    r_s, lr_s = primed("sgd", kappa, ETA["sgd"])
    rates, margins = [], []
    for s in range(seeds):
        layout, plan = adam_plan(priors, beta2, seed=s)
        co = discounts.softmax_psi_coefficients(jnp.asarray(B), jnp.asarray(O), priors, plan,
                                                BETA1, beta2, EPS, layout=layout)
        M = drift_matrix(co, O, r_a) * lr_a / ETA["adam"]
        rates.append(np.diag(M))
        margins.append(margin_rate(M))
    cm = discounts.softmax_moment_coefficients(jnp.asarray(B), jnp.asarray(O), priors,
                                               zs.moment_draws(len(priors)), EPS, BATCH,
                                               mode="sgd")
    Ms = drift_matrix(cm, O, r_s) * lr_s / ETA["sgd"]
    rates, margins = np.array(rates), np.array(margins)
    out = (rates.mean(0), rates.std(0, ddof=1) / np.sqrt(len(rates)), np.diag(Ms))
    if margin:
        out += (margins.mean(0), margins.std(0, ddof=1) / np.sqrt(len(margins)), margin_rate(Ms))
    return out


def run_frozen(args):
    OUT.joinpath("frozen").mkdir(parents=True, exist_ok=True)
    for kappa in args.kappas:
        for C in args.classes:
            for alpha in args.alphas:
                path = OUT / "frozen" / f"m0_k{kappa:g}_C{C}_a{alpha:g}.npz"
                if path.exists() and not args.overwrite:
                    continue
                zs = jax_modules()[0]
                priors = zs.zipf_priors(C, alpha)
                res = {"priors": priors}
                t0 = time.time()
                for beta2 in args.beta2s:
                    states = {"init": (B_START * kappa * np.eye(C), np.zeros((C, C)))}
                    run = Run("m0", "adam", kappa, C, alpha, beta2, ode_horizon("adam", C))
                    if ode_path(run).exists():
                        d = np.load(ode_path(run))
                        for t in (1.0, 10.0):
                            k = int(np.argmin(np.abs(d["time"] - t)))
                            states[f"t{t:g}"] = (d["B"][k], d["O"][k])
                    for name, (B, O) in states.items():
                        mean, se, sgd, m_a, m_se, m_s = frozen_rates(
                            priors, B, O, beta2=beta2, kappa=kappa, seeds=args.frozen_seeds,
                            margin=True)
                        res[f"{name}_b2{beta2:g}_adam"] = mean
                        res[f"{name}_b2{beta2:g}_adam_se"] = se
                        res[f"{name}_b2{beta2:g}_sgd"] = sgd
                        res[f"{name}_b2{beta2:g}_margin_adam"] = m_a
                        res[f"{name}_b2{beta2:g}_margin_adam_se"] = m_se
                        res[f"{name}_b2{beta2:g}_margin_sgd"] = m_s
                        res[f"{name}_b2{beta2:g}_O"] = np.asarray(O)
                np.savez_compressed(path, **res)
                print(f"frozen {path.name}: {time.time() - t0:.0f}s", flush=True)


def bias_chain_drift(priors, B, O, beta, *, beta2, kappa, chains, length, burn, seed=0):
    """Frozen-state bias drift by direct chains: ``E[m_k / sqrt(v_k + eps)]`` (Adam,
    ``b = 1``) and ``E f_k`` (SGD), with ``q | c ~ N(O[:, c] + beta, B)``; positive
    drift lowers the bias (``dbeta/dt = -eta * drift``)."""
    zs, jax, jnp, discounts, dynamics = jax_modules()
    from utils import sym_sqrt
    C = len(priors)
    root = sym_sqrt(jnp.asarray(B))
    mean = (jnp.asarray(O) + jnp.asarray(beta)[:, None]).T             # (C classes, C logits)
    logp = jnp.log(jnp.asarray(priors))

    def sample(key):
        k_c, k_z = jax.random.split(key)
        c = jax.random.categorical(k_c, logp, shape=(chains,))
        q = mean[c] + jax.random.normal(k_z, (chains, C)) @ root
        return jax.nn.softmax(q, axis=-1) - jax.nn.one_hot(c, C, dtype=q.dtype), c

    def step(carry, key):
        m, v, acc_u, acc_f, events = carry
        f, c = sample(key)
        m = BETA1 * m + (1 - BETA1) * f
        v = beta2 * v + (1 - beta2) * f ** 2
        return (m, v, acc_u, acc_f, events), (m / jnp.sqrt(v + EPS), f,
                                              jax.nn.one_hot(c, C, dtype=f.dtype))

    @jax.jit
    def run(key):
        keys = jax.random.split(key, burn + length)
        zero = jnp.zeros((chains, C))
        carry = (zero, zero, 0.0, 0.0, 0.0)
        carry, _ = jax.lax.scan(lambda c, k: (step(c, k)[0], None), carry, keys[:burn])

        def body(carry, key):
            carry, (u, f, hit) = step(carry, key)
            m, v, acc_u, acc_f, events = carry
            return (m, v, acc_u + u.sum(0), acc_f + f.sum(0), events + hit.sum(0)), None

        carry = (carry[0], carry[1], jnp.zeros(C), jnp.zeros(C), jnp.zeros(C))
        carry, _ = jax.lax.scan(body, carry, keys[burn:])
        return carry[2] / (chains * length), carry[3] / (chains * length), carry[4]

    # batch means over 8 independent blocks for standard errors
    blocks = [run(jax.random.fold_in(jax.random.PRNGKey(seed), j)) for j in range(8)]
    u = np.array([np.asarray(b[0]) for b in blocks])
    f = np.array([np.asarray(b[1]) for b in blocks])
    events = np.sum([np.asarray(b[2]) for b in blocks], axis=0)
    return {"adam": u.mean(0), "adam_se": u.std(0, ddof=1) / np.sqrt(8),
            "sgd": f.mean(0), "sgd_se": f.std(0, ddof=1) / np.sqrt(8), "events": events}


def run_frozen_bias(args):
    OUT.joinpath("frozen").mkdir(parents=True, exist_ok=True)
    zs = jax_modules()[0]
    for kappa in args.kappas:
        for C in args.classes:
            for alpha in args.alphas:
                path = OUT / "frozen" / f"m1bias_k{kappa:g}_C{C}_a{alpha:g}.npz"
                if path.exists() and not args.overwrite:
                    continue
                priors = zs.zipf_priors(C, alpha)
                logpi = np.log(priors)
                states = {"init": np.zeros(C), "prior": logpi - logpi.mean()}
                # >= 1000 own events of the rarest class per beta2 (8 blocks)
                length = int(np.ceil(1000 / (8 * args.bias_chains * priors[-1])))
                length = max(length, 4096)
                res = {"priors": priors, "length": length, "chains": args.bias_chains}
                t0 = time.time()
                for beta2 in args.beta2s:
                    burn = int(np.ceil(np.log(1e-8) / np.log(beta2)))
                    for name, beta in states.items():
                        out = bias_chain_drift(priors, B_START * kappa * np.eye(C),
                                               np.zeros((C, C)), beta, beta2=beta2,
                                               kappa=kappa, chains=args.bias_chains,
                                               length=length, burn=burn)
                        for k, v in out.items():
                            res[f"{name}_b2{beta2:g}_{k}"] = v
                np.savez_compressed(path, **res)
                print(f"frozen-bias {path.name}: {time.time() - t0:.0f}s (length {length})",
                      flush=True)


# --------------------------------------------------------------------------- #
# Subcommands
# --------------------------------------------------------------------------- #
def selected(args, kind):
    runs = matrix(models=args.models, kappas=args.kappas, classes=args.classes,
                  alphas=args.alphas, opts=args.opts, beta2s=args.beta2s, kind=kind,
                  etaset=args.etaset)
    if args.horizon is not None:
        runs = [r._replace(horizon=args.horizon) for r in runs]
    return runs


def run_ode(args):
    runs = [r for r in selected(args, "ode") if r.model == "m0"]
    for run in runs:
        run = run._replace(safety=args.safety)
        if args.dry_run:
            print(stem(run), "(cached)" if ode_path(run).exists() else "")
            continue
        ode_cached(run, extend=args.extend, overwrite=args.overwrite)


def run_sim(args):
    for run in selected(args, "sim"):
        if args.sim_horizon is not None:
            run = run._replace(horizon=args.sim_horizon)
        if args.dry_run:
            print(stem(run, args.dimension, "sim"),
                  "(cached)" if sim_path(run, args.dimension).exists() else "")
            continue
        sim_cached(run, args.dimension, args.runs, overwrite=args.overwrite)


def run_probe(args):
    """Wall time per simulated step (16 vmapped runs) and per ODE step."""
    zs, jax, jnp, discounts, dynamics = jax_modules()
    for C in args.classes:
        for model in ("m0", "m1"):
            for opt in ("adam", "sgd"):
                run = Run(model, opt, 1.0, C, 2.0, 0.99 if opt == "adam" else 0.0, 1.0)
                for n in (2_000, 20_000):
                    priors, data, theta0 = zs.zipf_model(C, 2.0, args.dimension)
                    steps = np.array([0, n // 2, n])
                    t0 = time.time()
                    out = zs.simulate_runs(priors, data, theta0, steps, lr=run.eta,
                                           beta1=BETA1, beta2=0.99, eps=EPS, batch=1,
                                           runs=args.runs, sgd=opt == "sgd",
                                           bias=model == "m1")
                    jax.block_until_ready(out)
                    dt = time.time() - t0
                print(f"probe sim C={C} {model} {opt} D={args.dimension}: "
                      f"{1e6 * dt / n:.1f} us/step ({args.runs} runs)", flush=True)


def run_tune(args):
    """Per-``kappa`` learning rates (ODE only), as E-series ``tune``: ``C = 32``,
    ``alpha = 1.5``, ``beta2 = 0.99``, criterion ``R(t = target)``; also reports
    ``R`` at ``t = 10`` and ``t = 1000`` and the time to ``R(0)/100``."""
    zs = jax_modules()[0]
    C, alpha = args.tune_classes, args.tune_alpha
    priors = zs.zipf_priors(C, alpha)
    T = 1e3
    times = grid(T)
    table = json.loads(ETA_FILE.read_text()) if ETA_FILE.exists() else {}
    results = {}
    for kappa in args.kappas:
        chosen = {}
        for opt, etas in (("adam", args.adam_etas), ("sgd", args.sgd_etas)):
            scores = []
            for eta in etas:
                t0 = time.time()
                out = solve_state(priors, B_START * kappa * np.eye(C), np.zeros((C, C)), times,
                                  opt=opt, beta2=0.99, kappa=kappa, safety=args.safety, eta=eta)
                loss, _ = zs.class_metrics(out["B"], out["O"])
                Rk = np.asarray(loss) @ priors
                at = lambda t: float(np.exp(np.interp(np.log(t), np.log(times[1:]),
                                                      np.log(Rk[1:]))))
                hit = zs.first_crossing(times, Rk, Rk[0] / 100)
                scores.append(at(args.tune_target))
                results[f"k{kappa:g}_{opt}_{eta:g}"] = Rk
                print(f"tune kappa={kappa:g} {opt} eta={eta:g}: R(10)={at(10):.3e}"
                      f" R({args.tune_target:g})={at(args.tune_target):.3e} R(1000)={at(1e3):.3e}"
                      f" t(R0/100)={hit:.3g}  [{time.time() - t0:.0f}s,"
                      f" {int(np.sum(out['steps']))} steps]", flush=True)
            k = int(np.nanargmin(scores))
            if k in (0, len(etas) - 1):
                print(f"  WARNING: {opt} optimum at the edge of the grid ({etas[k]:g})")
            chosen[opt] = float(etas[k])
        table[f"{kappa:g}"] = chosen
        print(f"tuned kappa={kappa:g}: {chosen}", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    ETA_FILE.write_text(json.dumps(table, indent=1))
    np.savez_compressed(OUT / "tune.npz", time=times, **results)


def run_selftest(args):
    sys.path.insert(0, str(ROOT / "experiments" / "imbalance"))
    import selftest
    selftest.main()


def run_numpy_script(name):
    sys.argv = [name]
    sys.path.insert(0, str(ANALYSIS))
    os.chdir(ANALYSIS)
    runpy.run_path(str(ANALYSIS / name), run_name="__main__")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=["selftest", "ode", "sim", "frozen", "frozen-bias",
                                       "probe", "tune", "analyze", "plot", "recheck",
                                       "offset"])
    p.add_argument("--models", nargs="+", default=list(MODELS))
    p.add_argument("--kappas", nargs="+", type=float, default=list(KAPPAS))
    p.add_argument("--classes", nargs="+", type=int, default=list(CLASSES))
    p.add_argument("--alphas", nargs="+", type=float, default=list(ALPHAS))
    p.add_argument("--opts", nargs="+", default=list(OPTS))
    p.add_argument("--beta2s", nargs="+", type=float, default=list(BETA2S))
    p.add_argument("--horizon", type=float, default=None)
    p.add_argument("--etaset", default="base", choices=["base", "tuned"])
    p.add_argument("--tune-classes", type=int, default=32)
    p.add_argument("--tune-alpha", type=float, default=1.5)
    p.add_argument("--tune-target", type=float, default=100.0)
    p.add_argument("--adam-etas", nargs="+", type=float, default=[0.1, 0.3, 1.0, 3.0])
    p.add_argument("--sgd-etas", nargs="+", type=float, default=[0.3, 1.0, 3.0, 10.0, 30.0])
    p.add_argument("--sim-horizon", type=float, default=None)
    p.add_argument("--safety", type=float, default=SAFETY)
    p.add_argument("--extend", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--dimension", type=int, default=1000)
    p.add_argument("--runs", type=int, default=16)
    p.add_argument("--frozen-seeds", type=int, default=4)
    p.add_argument("--bias-chains", type=int, default=4096)
    args = p.parse_args()
    if args.command in ("analyze", "plot", "recheck", "offset"):
        run_numpy_script({"analyze": "analyze.py", "plot": "figures.py",
                          "recheck": "recheck.py", "offset": "offset.py"}[args.command])
        return
    {"selftest": run_selftest, "ode": run_ode, "sim": run_sim, "frozen": run_frozen,
     "frozen-bias": run_frozen_bias, "probe": run_probe,
     "tune": run_tune}[args.command](args)


if __name__ == "__main__":
    main()
