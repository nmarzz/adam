"""Diagnostic: does the current (diagonal-K) diffusion match ground truth for a
DENSE covariance K?

Ground truth = the Adam *optimizer* (simulate.build_adam), which samples real data
x ~ N(0,K) and is correct for any K. We compare its risk trajectory against the
SDE and ODE, which both use the simplified sigma-hat (Remark 1). We pick a regime
(largish lr, long T) where the O(lr^2) diffusion term sets the risk floor, so a
wrong diffusion is visible.
"""
import config
config.enable_x64()

import jax
import jax.numpy as jnp

import problems
import simulate
import dynamics

prob = problems.get_problem("linreg")
d, m = 64, 1
T = 6
lr = 2.5                      # large-ish so the lr^2 diffusion term matters
beta1, beta2 = 0.1, 0.1
n_seeds = 64


def init(key):
    k0, ks = jax.random.split(key)
    theta0 = jax.random.normal(k0, (d, m)) / jnp.sqrt(d)
    star = jax.random.normal(ks, (d, m))
    star = star / jnp.linalg.norm(star, axis=0) * 3.0
    return theta0, star


def floor_compare(label, cov):
    theta0, star = init(jax.random.PRNGKey(0))
    sim = simulate.build_adam(prob, cov, star, lr / d, beta1=beta1, beta2=beta2)
    keys = jax.random.split(jax.random.PRNGKey(1), n_seeds)
    opt = simulate.run_many(sim, prob, theta0, star, cov, T * d, keys=keys).mean(0)

    ode, _ = dynamics.run_adam_ode(prob, theta0, star, cov, T, lr, beta1=beta1,
                                   beta2=beta2, dt=0.02, num_samples=20_000,
                                   key=jax.random.PRNGKey(2))
    sde, _ = dynamics.run_adam_sde(prob, theta0, star, cov, T, lr, beta1=beta1,
                                   beta2=beta2, dt=0.01, num_samples=20_000,
                                   key=jax.random.PRNGKey(3))
    # average the last 20% (the noise floor)
    f_opt = float(opt[int(0.8 * len(opt)):].mean())
    f_ode = float(ode[int(0.8 * len(ode)):].mean())
    f_sde = float(sde[int(0.8 * len(sde)):].mean())
    print(f"\n{label}: risk floor (last 20% mean)")
    print(f"  optimizer (truth) = {f_opt:.4f}")
    print(f"  ODE (sigma-hat)   = {f_ode:.4f}   rel err {abs(f_ode-f_opt)/f_opt:6.1%}")
    print(f"  SDE (sigma-hat)   = {f_sde:.4f}   rel err {abs(f_sde-f_opt)/f_opt:6.1%}")


if __name__ == "__main__":
    cov_diag = jnp.array([j ** -0.5 for j in range(1, d + 1)])
    floor_compare("DIAGONAL K", cov_diag)

    # dense K with the same eigenvalues but a random rotation
    A = jax.random.normal(jax.random.PRNGKey(9), (d, d))
    Qmat, _ = jnp.linalg.qr(A)
    K_dense = (Qmat * cov_diag) @ Qmat.T
    K_dense = (K_dense + K_dense.T) / 2
    floor_compare("DENSE K (rotated, same spectrum)", K_dense)
