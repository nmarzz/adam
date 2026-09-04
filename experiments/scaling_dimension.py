"""Dimension-scaling figure.

Hold the *risk* fixed while the data covariance scales in a natural way, then
show that the finite-dimensional Adam simulator concentrates onto the
d-independent HAdam SDE / ODE limit as the ambient dimension grows.

* Covariance: isotropic, ``Sigma = I``. Together with the fixed initial order
  parameter, this makes the deterministic ODE path dimension-independent and
  leaves finite-dimensional concentration as the only changing effect.
* Initial conditions: student and teacher are constructed in whitened
  coordinates with fixed Sigma-norms and fixed overlap. Consequently
  ``risk(theta0) = ||theta0-theta*||_Sigma^2 / 2`` is exactly the same for every
  dimension, rather than only converging to a common value as ``d`` grows.

Run: ``venv/bin/python experiments/scaling_dimension.py``
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
config.enable_x64()

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

import dynamics
import problems
import simulate
from utils import compute_ci

# ----------------------------- experiment knobs ---------------------------- #
PROBLEM = "linreg"
DIMS = [128, 256, 512]
N_SEEDS = 12
T = 2.0
SDE_LR = 0.7                 # learning rate in continuous-time units
BETA1, BETA2 = 0.1, 0.1
NORM0, NORMSTAR = 9.0, 1.0   # fixed squared Sigma-norms
INITIAL_OVERLAP = 0.0         # Sigma cosine between theta0 and theta*
ODE_DIM = 512                 # dimension at which to draw the limiting ODE curve
NUM_SAMPLES = 20_000


def fixed_spectrum(d):
    """Isotropic covariance for a clean dimension-scaling comparison."""
    return jnp.ones(d)


def fixed_risk_init(d, key):
    """Construct student/teacher with exactly fixed Sigma geometry."""
    if not -1.0 <= INITIAL_OVERLAP <= 1.0:
        raise ValueError("INITIAL_OVERLAP must lie in [-1, 1]")

    cov = fixed_spectrum(d)
    k0, ks = jax.random.split(key)

    # Euclidean geometry in whitened coordinates is Sigma geometry in the
    # original coordinates. Orthogonalize explicitly so the cross term cannot
    # fluctuate with dimension.
    star_direction = jax.random.normal(ks, (d, 1))
    star_direction /= jnp.linalg.norm(star_direction)
    orthogonal = jax.random.normal(k0, (d, 1))
    orthogonal -= star_direction * jnp.sum(star_direction * orthogonal)
    orthogonal /= jnp.linalg.norm(orthogonal)

    star_white = jnp.sqrt(NORMSTAR) * star_direction
    theta0_white = jnp.sqrt(NORM0) * (
        INITIAL_OVERLAP * star_direction
        + jnp.sqrt(1.0 - INITIAL_OVERLAP**2) * orthogonal
    )
    scale = jnp.sqrt(cov)[:, None]
    return theta0_white / scale, star_white / scale, cov


def main():
    prob = problems.get_problem(PROBLEM)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    colors = plt.cm.viridis(jnp.linspace(0.15, 0.8, len(DIMS)))

    for color, d in zip(colors, DIMS):
        theta0, star, cov = fixed_risk_init(d, jax.random.PRNGKey(d))
        sim = simulate.build_adam(
            prob, cov, star, SDE_LR / d, beta1=BETA1, beta2=BETA2
        )
        keys = jax.random.split(jax.random.PRNGKey(100 + d), N_SEEDS)
        risks = simulate.run_many(
            sim, prob, theta0, star, cov, int(T * d), keys=keys
        )
        t = jnp.arange(int(T * d)) / d
        mean, lo, hi = compute_ci(risks)
        ax.plot(t, mean, color=color, lw=1.8, label=f"Adam sim, d={d}")
        ax.fill_between(t, lo, hi, color=color, alpha=0.18)
        print(f"d={d:4d}: R0={float(mean[0]):.4f} RT={float(mean[-1]):.4f}")

    theta0, star, cov = fixed_risk_init(ODE_DIM, jax.random.PRNGKey(ODE_DIM))
    ode_risk, ode_t = dynamics.run_adam_ode(
        prob, theta0, star, cov, T, SDE_LR,
        beta1=BETA1, beta2=BETA2, dt=0.02,
        num_samples=NUM_SAMPLES, key=jax.random.PRNGKey(0),
    )
    ax.plot(ode_t, ode_risk, "k--", lw=2.0, label="HAdam ODE limit")
    print(f"ODE : R0={float(ode_risk[0]):.4f} RT={float(ode_risk[-1]):.4f}")

    ax.set_xlabel("rescaled time  t = k / d")
    ax.set_ylabel("risk")
    ax.set_yscale("log")
    ax.set_title(
        f"Adam: finite-d simulator -> HAdam limit ({PROBLEM}, "
        f"$\\beta_1$={BETA1}, $\\beta_2$={BETA2})"
    )
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()

    out = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figs"
    )
    fig.savefig(os.path.join(out, "scaling_dimension.png"), dpi=150)
    fig.savefig(os.path.join(out, "scaling_dimension.pdf"))
    print(f"saved -> {out}/scaling_dimension.{{png,pdf}}")


if __name__ == "__main__":
    main()
