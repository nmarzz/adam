"""Checks for the Zipf-scaling machinery (experiments/zipf_adam_plan.md, Phase 0).

Run ``real/bin/python tests/test_zipf_coefficients.py`` from the repository root
(or ``experiments/zipf_scaling.py selftest``). Every check prints PASS/FAIL and a
number; Monte-Carlo comparisons use several independent seeds and z-scores.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

import config
config.enable_x64()

import jax
import jax.numpy as jnp
import numpy as np

import discounts
import dynamics
import problems
import simulate

RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append(bool(ok))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name} {detail}", flush=True)


def zscores(a, b):
    a, b = np.asarray(a), np.asarray(b)
    se = np.sqrt(a.var(0, ddof=1) / len(a) + b.var(0, ddof=1) / len(b))
    return (a.mean(0) - b.mean(0)) / se


def zipf_state(C, alpha, D=2000, scale=2.0, seed=0):
    pi = np.arange(1, C + 1, dtype=float) ** -alpha
    pi /= pi.sum()
    Q, _ = jnp.linalg.qr(jax.random.normal(jax.random.PRNGKey(seed), (D, C)))
    means = 5.0 * Q
    theta = (0.1 * jax.random.normal(jax.random.PRNGKey(seed + 1), (D, C)) / jnp.sqrt(D)
             + scale * means / 25.0)
    return pi, theta.T @ theta, theta.T @ means


def test_matches_stationary_coefficients(seeds=6):
    """New softmax estimator == stationary_coefficients (C = 5, b in {1, 16})."""
    from multiclass_batch import PRIORS, mixture
    data, theta0 = mixture(2000, seed=3)
    lam = data.factor ** 2
    theta = theta0 + 0.5 * (data.means / jnp.sqrt(lam)[:, None])
    B, O = theta.T @ (lam[:, None] * theta), theta.T @ data.means
    f = problems.make_gmm_softmax(PRIORS).f
    pri = jnp.asarray(PRIORS)
    for b in (1, 16):
        layout0 = discounts.path_layout(0.9, 0.95, 1024)
        old, new, old_sgd, new_sgd = [], [], [], []
        for s in range(seeds):
            draws = discounts.draw_paths(jax.random.PRNGKey(100 + s), 128 if b == 1 else 32,
                                         layout0, batch=b, inputs=5, outputs=5)
            kw = dict(burn=layout0[0], tail=layout0[2])
            o = discounts.stationary_coefficients(B, O, pri, draws, f, 0.9, 0.95, 1e-2, **kw)
            og = discounts.stationary_coefficients(B, O, pri, draws, f, 0.9, 0.95, 1e-2,
                                                   sgd=True, **kw)
            layout, plan = discounts.softmax_plan(jax.random.PRNGKey(200 + s), PRIORS, 0.9,
                                                  0.95, batch=b,
                                                  plain_batches=2 ** 17 if b == 1 else 2 ** 15)
            n = discounts.softmax_coefficients(B, O, pri, plan, 0.9, 0.95, 1e-2, layout=layout)
            ng = discounts.softmax_coefficients(B, O, pri, plan, 0.9, 0.95, 1e-2,
                                                layout=layout, mode="sgd")
            for lst, co in ((old, o), (new, n), (old_sgd, og), (new_sgd, ng)):
                lst.append(np.concatenate([np.ravel(co[k]) for k in ("phi", "chi", "sigma")]))
        for tag, a, c in (("adam", new, old), ("sgd", new_sgd, old_sgd)):
            z = zscores(a, c)
            rms = float(np.sqrt(np.mean(z ** 2)))
            check(f"softmax_coefficients ({tag}) vs stationary_coefficients, b={b}",
                  rms < 1.6 and np.abs(z).max() < 5,
                  f"rms z {rms:.2f}, max |z| {np.abs(z).max():.2f} over {z.size} entries")


def test_forced_matches_plain(seeds=8):
    """Forced-class events reproduce plain events (all 8 classes forced)."""
    pi, B, O = zipf_state(8, 1.5)
    pri = jnp.asarray(pi)
    stat = lambda co: np.array([np.mean(np.diag(np.asarray(co[k])))
                                for k in ("phi", "chi", "sigma")])
    for b in (1, 16):
        plain, forced = [], []
        for s in range(seeds):
            lp, pp = discounts.softmax_plan(jax.random.PRNGKey(1000 + s), pi, 0.9, 0.95,
                                            batch=b, plain_batches=2 ** 16 if b == 1 else 2 ** 14,
                                            min_events=0)
            lf, pf = discounts.softmax_plan(jax.random.PRNGKey(2000 + s), pi, 0.9, 0.95,
                                            batch=b, plain_batches=1024, min_events=8000)
            plain.append(stat(discounts.softmax_coefficients(B, O, pri, pp, 0.9, 0.95, 1e-2,
                                                             layout=lp)))
            forced.append(stat(discounts.softmax_coefficients(B, O, pri, pf, 0.9, 0.95, 1e-2,
                                                              layout=lf)))
        z = zscores(forced, plain)
        rel = np.mean(forced, 0) / np.mean(plain, 0) - 1
        check(f"forced events vs plain events, b={b}", np.abs(z).max() < 3.5,
              f"class-mean diag rel diff (phi, chi, sigma) {np.round(rel, 4)}, z {np.round(z, 2)}")


def test_psgd_is_scaled_sgd():
    pi, B, O = zipf_state(8, 1.5)
    layout, plan = discounts.softmax_plan(jax.random.PRNGKey(3), pi, 0.9, 0.95, batch=4)
    kw = dict(layout=layout)
    sgd = discounts.softmax_coefficients(B, O, jnp.asarray(pi), plan, 0.9, 0.95, 1e-3,
                                         mode="sgd", **kw)
    psgd = discounts.softmax_coefficients(B, O, jnp.asarray(pi), plan, 0.9, 0.95, 1e-3,
                                          mode="psgd", **kw)
    scale = 1 / np.sqrt(np.diag(np.asarray(sgd["sigma"])) + 1e-3)
    err = max(np.abs(np.asarray(psgd["phi"]) - np.asarray(sgd["phi"]) * scale).max(),
              np.abs(np.asarray(psgd["chi"]) - np.asarray(sgd["chi"]) * scale).max())
    check("psgd = sgd with preconditioner (E f^2/b + eps)^-1/2", err < 1e-12, f"max err {err:.1e}")


def test_isotropic_collapse():
    """Aggregated K = Id state == mode-resolved ODE (D = 500)."""
    from zipf_scaling import zipf_model
    priors, data, theta0 = zipf_model(5, 1.5, 500)
    problem = problems.make_gmm_softmax(np.asarray(priors))
    layout = discounts.path_layout(0.9, 0.95, 512)
    draws = discounts.draw_paths(jax.random.PRNGKey(0), 64, layout, batch=1, inputs=5, outputs=5)
    cov = simulate.mixture_covariance(data)
    kw = dict(beta1=0.9, beta2=0.95, eps=1e-2, draws=draws, layout=layout, dimension=500,
              dt=0.02, record_every=25)
    modes, _ = dynamics.adam_modes(cov, theta0, data.teacher, data.means)
    full = dynamics.run_mixture_ode(problem, modes, 0.3, 3.0, **kw)
    agg_modes, initial = dynamics.isotropic_modes(theta0, data.teacher, data.means)
    agg = dynamics.run_mixture_ode(problem, agg_modes, 0.3, 3.0, initial=initial, **kw)
    err = max(float(jnp.abs(full["B"] - agg["B"]).max() / jnp.abs(full["B"]).max()),
              float(jnp.abs(full["O"] - agg["O"]).max() / jnp.abs(full["O"]).max()))
    check("collapsed isotropic ODE == mode-resolved ODE (D=500)", err < 1e-10,
          f"max relative difference {err:.1e}")
    return full, problem


def test_logtime_ode(full=None, problem=None):
    """Adaptive log-time integrator (new estimator) vs fixed-step ODE (old estimator)."""
    from zipf_scaling import solve, zipf_model
    priors, data, theta0 = zipf_model(5, 1.5, 500)
    if full is None:
        full, problem = test_isotropic_collapse()
    times = np.asarray(full["time"])
    kw = dict(lr=0.3, beta1=0.9, beta2=0.95, eps=1e-2, batch=1)
    coarse = solve(priors, theta0, data.means, times, safety=0.2, **kw)
    fine = solve(priors, theta0, data.means, times, safety=0.05, **kw)
    risk = lambda out: np.asarray(jax.vmap(problem.risk)(out["B"], out["O"]))
    r_full, r_coarse, r_fine = np.asarray(full["risk"]), risk(coarse), risk(fine)
    check("log-time ODE: safety 0.2 vs 0.05", np.abs(r_coarse - r_fine).max() < 1e-3,
          f"max |risk diff| {np.abs(r_coarse - r_fine).max():.1e},"
          f" steps {int(np.sum(coarse['steps']))} vs {int(np.sum(fine['steps']))}")
    rel = np.abs(r_coarse / r_full - 1).max()
    check("log-time ODE (softmax_coefficients) vs run_mixture_ode (stationary_coefficients)",
          rel < 0.01, f"max relative risk difference {rel:.4f} (MC level ~0.5%)")


def test_simulator_grid():
    from zipf_scaling import zipf_model
    priors, data, theta0 = zipf_model(5, 1.5, 300)
    problem = problems.make_gmm_softmax(np.asarray(priors))
    sim = simulate.build_mixture_adam(problem, data, 0.3 / 300, beta1=0.9, beta2=0.95,
                                      eps=1e-2, batch=2)
    key = jax.random.PRNGKey(4)
    B1, O1, _ = simulate.run_mixture(sim, data, theta0, 900, record_every=300, key=key)
    B2, O2 = simulate.run_mixture_at(sim, data, theta0, jnp.array([0, 300, 600, 900]), key=key)
    err = float(jnp.abs(B1 - B2).max() + jnp.abs(O1 - O2).max())
    check("run_mixture_at == run_mixture at shared record steps", err < 1e-12,
          f"max abs diff {err:.1e}")


def main():
    t0 = time.time()
    print("Zipf-scaling self-tests")
    test_matches_stationary_coefficients()
    test_forced_matches_plain()
    test_psgd_is_scaled_sgd()
    full, problem = test_isotropic_collapse()
    test_logtime_ode(full, problem)
    test_simulator_grid()
    print(f"{sum(RESULTS)}/{len(RESULTS)} passed in {time.time() - t0:.0f}s")
    return all(RESULTS)


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
