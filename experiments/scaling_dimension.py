"""Dimension-scaling figure.

Hold the *risk* fixed while the data covariance scales in a natural way, then show
that the finite-dimensional Adam simulator concentrates onto the d-independent
HAdam SDE / ODE limit as the ambient dimension grows.

* Covariance: diagonal with eigenvalues placed by a **fixed quantile function**
  ``lambda_j = q((j-0.5)/d)`` normalized to unit mean. The empirical spectral
  distribution is therefore the *same measure* for every ``d``.
* Initial conditions: ``theta0, theta* ~ N(0, I/d)`` rescaled in the Sigma-norm so
  ``||theta0||_Sigma^2`` and ``||theta*||_Sigma^2`` are fixed -> the initial risk
  is fixed across ``d`` (for linreg, risk = 1/2 ||theta0 - theta*||_Sigma^2, and
  the cross term vanishes as d grows).

Run:  ``venv/bin/python experiments/scaling_dimension.py``
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
config.enable_x64()

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

import problems
import simulate
import dynamics
from utils import make_B, compute_ci

# ----------------------------- experiment knobs ---------------------------- #
PROBLEM = "linreg"
DIMS = [128, 256, 512]
N_SEEDS = 12
T = 2.0
SDE_LR = 0.7                 # learning rate in continuous-time units
BETA1, BETA2 = 0.1, 0.1
NORM0, NORMSTAR = 9.0, 1.0   # fixed Sigma-norms of theta0 and theta*
ODE_DIM = 512               # dimension at which to draw the limiting ODE curve
NUM_SAMPLES = 20_000


def fixed_spectrum(d):
    """Diagonal Sigma from a fixed quantile function, normalized to unit mean."""
    u = (jnp.arange(1, d + 1) - 0.5) / d
    lam = 0.1 + u                      # quantile function q(u) = 0.1 + u on (0,1)
    return lam / lam.mean()


def fixed_risk_init(d, key):
    """theta0, theta* with prescribed Sigma-norms so the initial risk is fixed."""
    cov = fixed_spectrum(d)
    k0, ks = jax.random.split(key)
    theta0 = jax.random.normal(k0, (d, 1)) / jnp.sqrt(d)
    star = jax.random.normal(ks, (d, 1)) / jnp.sqrt(d)
    theta0 *= jnp.sqrt(NORM0 / (theta0.T @ (cov[:, None] * theta0)))
    star *= jnp.sqrt(NORMSTAR / (star.T @ (cov[:, None] * star)))
    return theta0, star, cov


def main():
    prob = problems.get_problem(PROBLEM)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    colors = plt.cm.viridis(jnp.linspace(0.15, 0.8, len(DIMS)))

    for color, d in zip(colors, DIMS):
        theta0, star, cov = fixed_risk_init(d, jax.random.PRNGKey(d))
        sim = simulate.build_adam(prob, cov, star, SDE_LR / d, beta1=BETA1, beta2=BETA2)
        keys = jax.random.split(jax.random.PRNGKey(100 + d), N_SEEDS)
        risks = simulate.run_many(sim, prob, theta0, star, cov, int(T * d), keys=keys)
        t = jnp.arange(int(T * d)) / d
        mean, lo, hi = compute_ci(risks)
        ax.plot(t, mean, color=color, lw=1.8, label=f"Adam sim, d={d}")
        ax.fill_between(t, lo, hi, color=color, alpha=0.18)
        print(f"d={d:4d}: R0={float(mean[0]):.4f} RT={float(mean[-1]):.4f}")

    # d-independent limit curves at ODE_DIM
    theta0, star, cov = fixed_risk_init(ODE_DIM, jax.random.PRNGKey(ODE_DIM))
    ode_risk, ode_t = dynamics.run_adam_ode(
        prob, theta0, star, cov, T, SDE_LR, beta1=BETA1, beta2=BETA2,
        dt=0.02, num_samples=NUM_SAMPLES, key=jax.random.PRNGKey(0))
    ax.plot(ode_t, ode_risk, "k--", lw=2.0, label="HAdam ODE limit")
    print(f"ODE : R0={float(ode_risk[0]):.4f} RT={float(ode_risk[-1]):.4f}")

    ax.set_xlabel("rescaled time  t = k / d")
    ax.set_ylabel("risk")
    ax.set_yscale("log")
    ax.set_title(f"Adam: finite-d simulator -> HAdam limit ({PROBLEM}, "
                 f"$\\beta_1$={BETA1}, $\\beta_2$={BETA2})")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()

    out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figs")
    fig.savefig(os.path.join(out, "scaling_dimension.png"), dpi=150)
    fig.savefig(os.path.join(out, "scaling_dimension.pdf"))
    print(f"saved -> {out}/scaling_dimension.{{png,pdf}}")


if __name__ == "__main__":
    main()
