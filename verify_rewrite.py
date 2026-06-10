"""Self-contained smoke + correctness checks for the simulators.

Run from the ``code/`` directory:  ``venv/bin/python verify_rewrite.py``
Everything runs at small ``d`` (<= 256) so it is fast.
"""
import time

import config
config.enable_x64()  # before any array creation

import jax
import jax.numpy as jnp

import problems
import simulate
import dynamics

PASS, FAIL = "PASS", "FAIL"


def check(name, ok, extra=""):
    print(f"  [{PASS if ok else FAIL}] {name} {extra}")
    return ok


def setup(d, m=1, seed=0):
    key = jax.random.PRNGKey(seed)
    k1, k2 = jax.random.split(key)
    params0 = jax.random.normal(k1, (d, m)) / jnp.sqrt(d)
    optimal = jax.random.normal(k2, (d, m))
    optimal = optimal / jnp.linalg.norm(optimal, axis=0) * 5.0
    cov = jnp.array([j ** -0.5 for j in range(1, d + 1)])
    return params0, optimal, cov


def dense_cov(d, seed=9):
    cov = jnp.array([j ** -0.5 for j in range(1, d + 1)])
    Qm, _ = jnp.linalg.qr(jax.random.normal(jax.random.PRNGKey(seed), (d, d)))
    K = (Qm * cov) @ Qm.T
    return (K + K.T) / 2


def test_cov_vectorization():
    """The einsum identity in cov_from_B must equal an explicit per-l loop."""
    print("cov_from_B vectorization == explicit loop")
    key = jax.random.PRNGKey(3)
    f = problems._f_linreg
    B = jnp.array([[2.0, 0.7], [0.7, 1.5]])
    beta1, beta2 = 0.3, 0.4
    H, ns = 8, 4000

    kQ, kQh, kz, kzh = jax.random.split(key, 4)
    Q = jax.random.multivariate_normal(kQ, jnp.zeros(2), B, shape=(ns, 1))
    z = jax.random.normal(kz, (ns, 1))
    Qh = jax.random.multivariate_normal(kQh, jnp.zeros(2), B, shape=(ns, H))
    zh = jax.random.normal(kzh, (ns, H))
    d1 = beta1 ** jnp.arange(H) * (1 - beta1)
    d2 = beta2 ** jnp.arange(H) * (1 - beta2)
    fq = f(Q).squeeze(1)
    cur = fq * z
    cur2 = cur ** 2
    base2 = f(Qh) ** 2 * zh[:, :, None] ** 2

    sms = []
    for l in range(H):
        sg = base2.at[:, l, :].set(cur2)
        sms.append(jnp.sqrt(jnp.einsum("nlm,l->nm", sg, d2)))
    sms = jnp.stack(sms, axis=1)
    upd_ref = jnp.einsum("l,nlm->nm", d1, cur[:, None, :] / sms)
    ref = jnp.einsum("nm,nk->mk", upd_ref, upd_ref) / ns

    S = jnp.einsum("h,nhm->nm", d2, base2)
    sm = jnp.sqrt(S[:, None, :] + d2[None, :, None] * (cur2[:, None, :] - base2))
    upd = jnp.einsum("h,nhm->nm", d1, cur[:, None, :] / sm)
    vec = jnp.einsum("nm,nk->mk", upd, upd) / ns

    diff = float(jnp.max(jnp.abs(ref - vec)))
    return check("max|Δ|", diff < 1e-9, f"={diff:.2e}")


def test_optimizers_run():
    print("optimizers run, finite, risk decreases")
    d = 128
    p0, opt, cov = setup(d)
    prob = problems.get_problem("linreg")
    T, lr = 1, 0.7 / d
    key = jax.random.PRNGKey(7)
    sims = {
        "adam": simulate.build_adam(prob, cov, opt, lr, beta1=0.1, beta2=0.1),
        "block_adam": simulate.build_block_adam(prob, cov, opt, lr, beta1=0.1, beta2=0.1),
        "sgd": simulate.build_sgd(prob, cov, opt, lr),
        "block_sgd": simulate.build_block_sgd(prob, cov, opt, lr),
        "resampled_adam": simulate.build_resampled_adam(prob, cov, opt, lr, beta1=0.1, beta2=0.1),
    }
    ok = True
    for nm, sim in sims.items():
        _, risks = simulate.run(sim, prob, p0, opt, cov, T * d, key=key)
        finite = bool(jnp.all(jnp.isfinite(risks)))
        decreased = float(risks[-1]) < float(risks[0])
        ok &= check(nm, finite and decreased,
                    f"R0={float(risks[0]):.3f} -> RT={float(risks[-1]):.3f}")
    return ok


def test_block_adam_tracks_adam():
    print("Block Adam tracks Adam (d=256, mean over seeds)")
    d = 256
    p0, opt, cov = setup(d)
    prob = problems.get_problem("linreg")
    T, lr = 1, 0.7 / d
    keys = jax.random.split(jax.random.PRNGKey(11), 8)
    adam = simulate.build_adam(prob, cov, opt, lr, beta1=0.1, beta2=0.1)
    badam = simulate.build_block_adam(prob, cov, opt, lr, beta1=0.1, beta2=0.1)
    r_adam = simulate.run_many(adam, prob, p0, opt, cov, T * d, keys=keys).mean(0)
    r_badam = simulate.run_many(badam, prob, p0, opt, cov, T * d, keys=keys).mean(0)
    rel = float(jnp.abs(r_adam[-1] - r_badam[-1]) / r_adam[-1])
    return check("final relative gap", rel < 0.15, f"={rel:.3f}")


def test_dynamics_run():
    print("ODE / SDE integrators run and stay finite (diagonal + dense K)")
    d = 128
    p0, opt, cov = setup(d)
    K = dense_cov(d)
    prob = problems.get_problem("linreg")
    key = jax.random.PRNGKey(5)
    ok = True
    r, _ = dynamics.run_adam_ode(prob, p0, opt, cov, 1, 0.7, beta1=0.1, beta2=0.1,
                                 dt=0.05, num_samples=5000, key=key)
    ok &= check("adam_ode (diag)", bool(jnp.all(jnp.isfinite(r))))
    r, _ = dynamics.run_sgd_ode(prob, p0, opt, cov, 1, 0.7, dt=0.05, key=key)
    ok &= check("sgd_ode (diag)", bool(jnp.all(jnp.isfinite(r))))
    r, _ = dynamics.run_adam_sde(prob, p0, opt, cov, 0.2, 0.7, beta1=0.1, beta2=0.1,
                                 dt=0.02, num_samples=5000, key=key)
    ok &= check("adam_sde (diag)", bool(jnp.all(jnp.isfinite(r))))
    # dense covariance: general Sigma path
    r, _ = dynamics.run_adam_ode(prob, p0, opt, K, 1, 1.5, beta1=0.1, beta2=0.1,
                                 dt=0.05, num_samples=5000, noise_samples=2000,
                                 noise_history=30, key=key)
    ok &= check("adam_ode (dense K)", bool(jnp.all(jnp.isfinite(r))))
    r, _ = dynamics.run_adam_sde(prob, p0, opt, K, 0.5, 1.5, beta1=0.1, beta2=0.1,
                                 dt=0.02, num_samples=5000, noise_samples=2000,
                                 noise_history=30, key=key)
    ok &= check("adam_sde (dense K)", bool(jnp.all(jnp.isfinite(r))))
    return ok


def benchmark():
    print("timing: scan-driven Adam, d=512, T=2")
    d = 512
    p0, opt, cov = setup(d)
    prob = problems.get_problem("linreg")
    T, lr = 2, 0.7 / d
    sim = simulate.build_adam(prob, cov, opt, lr, beta1=0.1, beta2=0.1)
    key = jax.random.PRNGKey(0)
    simulate.run(sim, prob, p0, opt, cov, T * d, key=key)[1].block_until_ready()  # warm jit
    t0 = time.time()
    simulate.run(sim, prob, p0, opt, cov, T * d, key=key)[1].block_until_ready()
    print(f"    {time.time() - t0:.3f}s for {T * d} steps")


if __name__ == "__main__":
    print(f"backend={config.default_backend()} devices={config.devices()}\n")
    results = []
    for fn in [test_cov_vectorization, test_optimizers_run,
               test_block_adam_tracks_adam, test_dynamics_run]:
        results.append(fn())
        print()
    benchmark()
    print(f"\n{'ALL PASSED' if all(results) else 'SOME FAILED'} "
          f"({sum(results)}/{len(results)})")
