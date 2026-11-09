"""Self-tests for series I (``real/bin/python experiments/imbalance.py selftest``).

1. ``bias=False`` simulation is bit-identical to the simulator at ``git HEAD~`` of
   ``simulate.py`` (the version before bias support); ``K = Id`` default unchanged.
2. Bias gradient: at ``beta = 0`` the weight rows equal the no-bias gradient, the bias
   row equals ``mean f``; SGD conserves ``sum(beta)``; Adam stability on the bias row is ``eps``.
3. Rescaling identity: ``K = kappa Id`` with ``eta`` equals ``K = Id`` with ``r / sqrt(kappa)``,
   ``eta'`` and ``Theta' = sqrt(kappa) Theta`` (Adam and SGD), to round-off. With a bias the
   identity holds only if the bias keeps ``eta`` (its gradient has no ``x``): checked with a
   per-row learning rate.
4. ODE extension is exact (run to 10, extend to 100, vs direct 100).
5. Regression: the series-I ODE reproduces the cached E2 C = 64 runs (Adam and SGD).
"""
from __future__ import annotations

import importlib.util
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

import imbalance as I  # noqa: E402

zs, jax, jnp, discounts, dynamics = I.jax_modules()
import problems  # noqa: E402
import simulate  # noqa: E402

RESULTS = []


def check(name, ok, detail):
    RESULTS.append((name, bool(ok), detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}", flush=True)


def old_simulate():
    """``simulate.py`` before the bias change (last commit touching it)."""
    src = subprocess.check_output(["git", "show", "HEAD:simulate.py"], cwd=ROOT, text=True)
    path = Path(tempfile.mkdtemp()) / "simulate_old.py"
    path.write_text(src)
    spec = importlib.util.spec_from_file_location("simulate_old", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_bit_identical():
    old = old_simulate()
    priors, data, theta0 = zs.zipf_model(8, 2.0, 300)
    problem = problems.make_gmm_softmax(np.asarray(priors))
    steps = np.array([0, 10, 100, 1000])
    keys = jax.random.split(jax.random.PRNGKey(3), 4)
    worst = 0.0
    for b in (1, 4):
        for mod_new, mod_old in ((simulate, old),):
            new_adam = mod_new.build_mixture_adam(problem, data, 0.3 / 300, beta1=0.9,
                                                  beta2=0.95, eps=1e-8, batch=b)
            old_data = old.MixtureData(*data)
            old_adam = mod_old.build_mixture_adam(problem, old_data, 0.3 / 300, beta1=0.9,
                                                  beta2=0.95, eps=1e-8, batch=b)
            n = mod_new.run_mixture_at_many(new_adam, data, theta0, steps, keys=keys)
            o = mod_old.run_mixture_at_many(old_adam, old_data, theta0, steps, keys=keys)
            worst = max(worst, max(float(jnp.max(jnp.abs(a - c))) for a, c in zip(n, o)))
            new_sgd = mod_new.build_mixture_sgd(problem, data, 3 / 300, batch=b)
            old_sgd = mod_old.build_mixture_sgd(problem, old_data, 3 / 300, batch=b)
            n = mod_new.run_mixture_at_many(new_sgd, data, theta0, steps, keys=keys)
            o = mod_old.run_mixture_at_many(old_sgd, old_data, theta0, steps, keys=keys)
            worst = max(worst, max(float(jnp.max(jnp.abs(a - c))) for a, c in zip(n, o)))
    check("bias=False bit-identical to previous simulator", worst == 0.0, f"max diff {worst:g}")


def test_bias_gradient():
    priors, data, theta0 = zs.zipf_model(8, 2.0, 300)
    problem = problems.make_gmm_softmax(np.asarray(priors))
    key = jax.random.PRNGKey(5)
    aug = jnp.concatenate([theta0, jnp.zeros((1, 8))])
    g0 = simulate._mixture_grad(problem, data, theta0, key, 4)
    g1 = simulate._mixture_grad(problem, data, aug, key, 4, bias=True)
    # bias row = mean f: recompute f from the weight-row gradient's sample
    k_c, k_z, k_w = jax.random.split(key, 3)
    c = jax.random.categorical(k_c, jnp.log(problem.priors), shape=(4,))
    x = data.means.T[c] + jax.random.normal(k_z, (4, 300)) * data.factor
    f = jax.nn.softmax(x @ theta0, axis=-1) - jax.nn.one_hot(c, 8)
    d_w = float(jnp.max(jnp.abs(g1[:-1] - g0)))
    d_b = float(jnp.max(jnp.abs(g1[-1] - f.mean(0))))
    check("bias gradient at beta = 0", d_w < 1e-14 and d_b < 1e-14,
          f"weight rows {d_w:.1e}, bias row vs mean f {d_b:.1e}")

    steps = np.array([0, 100, 3000])
    _, _, beta = zs.simulate_runs(priors, data, theta0, steps, lr=3.0, beta1=0.9, beta2=0.99,
                                  eps=1e-8, batch=2, runs=2, sgd=True, bias=True)
    s = float(jnp.max(jnp.abs(beta.sum(-1))))
    moved = float(jnp.max(jnp.abs(beta)))
    check("SGD conserves sum(beta)", s < 1e-10 and moved > 1e-3,
          f"max |sum beta| {s:.1e} (max |beta| {moved:.3f})")

    sim = simulate.build_mixture_adam(problem, data, 1.0, beta1=0.9, beta2=0.99, eps=0.37,
                                      batch=1, bias=True)
    # one step from m = v = 0 with g = 0 on a row gives update 0 / sqrt(stability):
    # check the stability array through a zero-gradient probe of the closure
    stab = None
    for cell in sim.step.__closure__:
        v = cell.cell_contents
        if isinstance(v, jnp.ndarray) and v.shape == (301, 1):
            stab = v
    ok = stab is not None and float(stab[-1, 0]) == 0.37 and float(stab[0, 0]) == 0.37
    check("Adam stability on the bias row is eps", ok,
          f"bias row {None if stab is None else float(stab[-1, 0])}")


def test_rescaling():
    kappa, D, C = 0.1, 400, 8
    priors, data, theta0 = zs.zipf_model(C, 2.0, D, noise=kappa)
    r_p = I.R / np.sqrt(kappa)
    data_p = simulate.MixtureData(jnp.ones(D), data.means / np.sqrt(kappa), data.teacher)
    theta_p = np.sqrt(kappa) * theta0
    steps = np.array([0, 50, 500, 4000])
    worst = {}
    for opt in ("adam", "sgd"):
        _, eta_p = I.primed(opt, kappa, I.ETA[opt])
        kw = dict(beta1=0.9, beta2=0.95, eps=1e-8, batch=2, runs=2, sgd=opt == "sgd", seed=4)
        a = zs.simulate_runs(priors, data, theta0, steps, lr=I.ETA[opt], **kw)
        b = zs.simulate_runs(priors, data_p, theta_p, steps, lr=eta_p, **kw)
        worst[opt] = max(float(jnp.max(jnp.abs(x - y)) / jnp.max(jnp.abs(x)))
                         for x, y in zip(a, b))
    ok = all(v < 1e-9 for v in worst.values())
    check("K = kappa Id equals primed K = Id problem (no bias; r' = r/sqrt(kappa))", ok,
          ", ".join(f"{k} {v:.1e}" for k, v in worst.items()) + f" (r' = {r_p:.2f})")
    # With a bias the identity needs the bias lr unscaled (its gradient f has no x):
    # in primed units the bias runs at eta, the weights at eta'. Check that the primed
    # problem reproduces K = kappa Id once the bias row's step is rescaled by eta/eta'.
    worst = {}
    for opt in ("adam", "sgd"):
        _, eta_p = I.primed(opt, kappa, I.ETA[opt])
        problem = problems.make_gmm_softmax(np.asarray(priors))
        aug = lambda th: jnp.concatenate([th, jnp.zeros((1, C))])
        D1 = D + 1
        if opt == "adam":
            mk = lambda dat, lr: simulate.build_mixture_adam(problem, dat, lr, beta1=0.9,
                                                             beta2=0.95, eps=1e-8, batch=2,
                                                             bias=True)
        else:
            mk = lambda dat, lr: simulate.build_mixture_sgd(problem, dat, lr, batch=2, bias=True)
        lr_rows = jnp.concatenate([jnp.full(D, eta_p / D), jnp.full(1, I.ETA[opt] / D)])[:, None]
        keys = jax.random.split(jax.random.PRNGKey(4), 2)
        a = simulate.run_mixture_at_many(mk(data, I.ETA[opt] / D), data, aug(theta0), steps,
                                         keys=keys, bias=True)
        b = simulate.run_mixture_at_many(mk(data_p, lr_rows), data_p, aug(theta_p), steps,
                                         keys=keys, bias=True)
        worst[opt] = max(float(jnp.max(jnp.abs(x - y)) / jnp.max(jnp.abs(x)))
                         for x, y in zip(a, b))
    ok = all(v < 1e-9 for v in worst.values())
    check("with bias: identity holds with the bias lr unscaled (eta), weights at eta'", ok,
          ", ".join(f"{k} {v:.1e}" for k, v in worst.items()))


def test_extend():
    old_out = I.OUT
    I.OUT = Path(tempfile.mkdtemp())
    try:
        worst = {}
        for opt, beta2 in (("adam", 0.95), ("sgd", 0.0)):
            for kappa in (1.0, 0.1):
                run = I.Run("m0", opt, kappa, 8, 2.0, beta2, 100.0)
                direct = I.ode_cached(run)
                I.ode_path(run).unlink()
                I.ode_cached(run._replace(horizon=10.0))
                ext = I.ode_cached(run, extend=True)
                worst[f"{opt} k{kappa:g}"] = float(np.max(np.abs(ext["loss"] - direct["loss"])))
        ok = all(v < 1e-10 for v in worst.values())
        check("ODE extension 10 -> 100 equals a direct run", ok,
              ", ".join(f"{k} {v:.1e}" for k, v in worst.items()))
    finally:
        I.OUT = old_out


def test_regression():
    cache = ROOT / "figs" / "experiments" / "zipf" / "ode"
    T = 100.0
    times = I.grid(T)
    worst = {}
    for opt, name in (("sgd", "sgd_C64_a2_b1_eta3_T1000_s1.npz"),
                      ("adam", "adam_C64_a2_b20.99_b1_e1e-08_eta0.3_T1000_s1.npz")):
        ref = np.load(cache / name)
        n = len(times)
        assert np.allclose(ref["time"][:n], times)
        priors = zs.zipf_priors(64, 2.0)
        _, data, _ = zs.zipf_model(64, 2.0, 4000)
        th = zs.deterministic_start(64, data.means)
        B0, O0 = th.T @ th, th.T @ data.means       # what E2 started from
        t0 = time.time()
        out = I.solve_state(priors, B0, O0, times, opt=opt, beta2=0.99, kappa=1.0, safety=1.0,
                            legacy_plan=True)    # E2 used the multi-chunk psi layout
        loss, _ = zs.class_metrics(out["B"], out["O"])
        worst[opt] = (float(np.max(np.abs(np.asarray(out["B"]) - ref["B"][:n]))),
                      float(np.max(np.abs(loss - ref["loss"][:n]))), time.time() - t0)
    ok = all(v[0] < 1e-9 and v[1] < 1e-9 for v in worst.values())
    check("regression vs cached E2 C = 64 runs (t <= 100)", ok,
          ", ".join(f"{k}: |dB| {v[0]:.1e}, |dloss| {v[1]:.1e} [{v[2]:.0f}s]"
                    for k, v in worst.items()))


def main():
    t0 = time.time()
    for test in (test_bit_identical, test_bias_gradient, test_rescaling, test_extend,
                 test_regression):
        try:
            test()
        except Exception as exc:  # report and continue
            check(test.__name__, False, f"error: {type(exc).__name__}: {exc}")
    n_pass = sum(ok for _, ok, _ in RESULTS)
    print(f"\n{n_pass}/{len(RESULTS)} passed in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
