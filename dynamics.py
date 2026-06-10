"""ODE and SDE integrators for the high-dimensional limit.

* The **ODEs** evolve the per-mode blocks ``y = [p, u, q]`` of the low-dim
  statistic ``B`` (one ``M+ x M+`` matrix per spectral mode) with forward Euler.
* The **SDEs** (HAdam / SGD) evolve the full ``D x M`` parameter with
  Euler–Maruyama.

Both are driven by ``lax.scan`` instead of the old Python ``for`` loops, and the
dense-covariance ODE initialization is vectorized (no ``for j in range(d)``).
Drift/diffusion come from :mod:`discounts`.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import lax

from discounts import (adam_noise_field, adam_ode_diffusion, compute_H, compute_I,
                       cov_from_B, phi_from_B)
from problems import Problem
from utils import make_B


def _lr_at(lr, t):
    return lr(t) if callable(lr) else lr


def _real_eig(mat):
    eigs, L = jnp.linalg.eig(mat)
    R = jnp.linalg.inv(L).T
    return jnp.real(eigs), jnp.real(L), jnp.real(R)


# --------------------------------------------------------------------------- #
# B reconstruction from ODE state
# --------------------------------------------------------------------------- #
def _make_B_adam(y, eigs):
    d = len(eigs)
    p, u, q = y[:d], y[d:2 * d], y[2 * d:]
    B11, B12, B22 = p.sum(0), u.sum(0), q.sum(0)
    return jnp.block([[B11, B12], [B12.T, B22]])


def _make_B_sgd(y, eigs):
    d = len(eigs)
    p, u, q = y[:d], y[d:2 * d], y[2 * d:]
    blocks = jnp.block([[p, u], [jnp.swapaxes(u, 1, 2), q]])
    return jnp.einsum("abc,a->bc", blocks, eigs)


# --------------------------------------------------------------------------- #
# ODE initialization (vectorized)
# --------------------------------------------------------------------------- #
def _modes(A, R, params, optimal_params):
    """Per-mode outer products ``einsum('aj,jb->jab', .T@A, R.T@.)``."""
    PA = params.T @ A          # (m, d)
    OA = optimal_params.T @ A
    RP = R.T @ params          # (d, m)
    RO = R.T @ optimal_params
    p = jnp.einsum("aj,jb->jab", PA, RP)
    u = jnp.einsum("aj,jb->jab", PA, RO)
    q = jnp.einsum("aj,jb->jab", OA, RO)
    return p, u, q


def _diag_modes(scale, params, optimal_params):
    """Per-mode outer products for diagonal covariance (mode == coordinate)."""
    p = jnp.einsum("jm,jn->jmn", params, params) * scale[:, None, None]
    u = jnp.einsum("jm,jn->jmn", params, optimal_params) * scale[:, None, None]
    q = jnp.einsum("jm,jn->jmn", optimal_params, optimal_params) * scale[:, None, None]
    return p, u, q


def _init_adam_ode(cov, params, optimal_params):
    """Returns (y0, eigs, diff_data). ``diff_data`` carries what the diffusion term
    needs: for diagonal ``K`` the per-mode scale ``var_force``; for dense ``K`` the
    eigenbasis pieces ``(KL, R, cov_chol)`` for the exact per-mode diffusion."""
    if cov.ndim == 1:                       # diagonal covariance
        eigs = jnp.sqrt(cov)
        p, u, q = _diag_modes(cov, params, optimal_params)
        diff_data = ("diag", cov)           # var_force == cov
    else:
        covbar = cov / jnp.sqrt(jnp.diag(cov))
        eigs, L, R = _real_eig(covbar)
        A = cov @ L                         # = K L
        p, u, q = _modes(A, R, params, optimal_params)
        diff_data = ("dense", A, R, jnp.linalg.cholesky(cov))
    return jnp.concatenate([p, u, q]), eigs, diff_data


def _init_sgd_ode(cov, params, optimal_params):
    if cov.ndim == 1:
        eigs = cov
        p, u, q = _diag_modes(jnp.ones_like(cov), params, optimal_params)
    else:
        eigs, L, R = _real_eig(cov)
        p, u, q = _modes(L, R, params, optimal_params)
    return jnp.concatenate([p, u, q]), eigs, None


# --------------------------------------------------------------------------- #
# ODE drivers
# --------------------------------------------------------------------------- #
def run_adam_ode(problem: Problem, params, optimal_params, cov, T, lr, *,
                 beta1, beta2, dt=0.01, num_samples=100_000, eps=0.0,
                 noise_samples=2000, noise_history=50, key):
    """HAdam ODE. The diffusion term handles a **general** covariance ``K``.

    * Diagonal ``K``: cheap ``var_force[j] * sigma-hat`` per mode (Remark 1).
    * Dense ``K``: exact per-mode diffusion from the full ``Sigma`` tensor via
      :func:`discounts.adam_ode_diffusion` (``noise_samples`` / ``noise_history``
      control its Monte-Carlo cost).
    """
    y0, eigs, diff_data = _init_adam_ode(cov, params, optimal_params)
    diagonal = diff_data[0] == "diag"
    d = len(eigs)
    iters = int(T / dt)

    def step(carry, i):
        y, key = carry
        B = _make_B_adam(y, eigs)
        risk = problem.risk_from_B(B)
        m = len(B) // 2
        key, k_phi, k_cov = jax.random.split(key, 3)
        phi = phi_from_B(B, problem.f, beta1, beta2, k_phi, num_samples=num_samples)
        phi1, phi2 = phi[:m], phi[m:]
        p, u, q = y[:d], y[d:2 * d], y[2 * d:]
        lr_t = _lr_at(lr, i * dt)
        e = eigs[:, None, None]

        if diagonal:
            var_force = diff_data[1]
            sigma = cov_from_B(B, problem.f, beta1, beta2, k_cov, num_samples=num_samples)
            diffusion = var_force[:, None, None] * sigma            # (D, M, M)
        else:
            _, KL, R, cov_chol = diff_data
            diffusion = adam_ode_diffusion(B, problem.f, beta1, beta2, k_cov, cov_chol,
                                           KL, R, num_samples=noise_samples,
                                           history_length=noise_history, eps=eps)

        p_up = -2 * lr_t * e * (p * phi1 + u * phi2) + lr_t ** 2 * diffusion / d
        u_up = -lr_t * e * (phi1 * u + phi2 * q)
        y = y + dt * jnp.concatenate([p_up, u_up, jnp.zeros_like(u_up)])
        return (y, key), risk

    (_, _), risks = lax.scan(step, (y0, key), jnp.arange(iters))
    return risks, jnp.arange(iters) * dt


def run_sgd_ode(problem: Problem, params, optimal_params, cov, T, lr, *,
                dt=0.01, key):
    y0, eigs, _ = _init_sgd_ode(cov, params, optimal_params)
    d = len(eigs)
    iters = int(T / dt)

    def step(carry, i):
        y, key = carry
        B = _make_B_sgd(y, eigs)
        risk = problem.risk_from_B(B)
        m = len(B) // 2
        key, k_h, k_i = jax.random.split(key, 3)
        H = compute_H(B, problem.f, k_h)
        I = compute_I(B, problem.f, k_i)
        p, u, q = y[:d], y[d:2 * d], y[2 * d:]
        lr_t = _lr_at(lr, i * dt)
        e = eigs[:, None, None]
        p_up = -2 * lr_t * e * (p * H[:m] + u * H[m:]) + e * lr_t ** 2 * I / d
        u_up = -lr_t * e * (H[:m] * u + H[m:] * q)
        y = y + dt * jnp.concatenate([p_up, u_up, jnp.zeros_like(u_up)])
        return (y, key), risk

    (_, _), risks = lax.scan(step, (y0, key), jnp.arange(iters))
    return risks, jnp.arange(iters) * dt


# --------------------------------------------------------------------------- #
# SDE drivers (full D x M parameter, Euler-Maruyama)
# --------------------------------------------------------------------------- #
def run_adam_sde(problem: Problem, params, optimal_params, cov, T, lr, *,
                 beta1, beta2, dt=0.005, num_samples=100_000, eps=0.0,
                 noise_samples=2000, noise_history=50, key):
    """HAdam SDE. The diffusion term handles a **general** covariance ``K``.

    * Diagonal ``K`` (``cov.ndim == 1``): uses the cheap ``sigma-hat ⊗ Id``
      simplification (Remark 1) — isotropic noise ``W @ sqrt(sigma-hat)``.
    * Dense ``K``: samples the full ``Sigma`` tensor via
      :func:`discounts.adam_noise_field`, so the ambient coordinates of the noise
      are correlated through ``K``. ``noise_samples`` / ``noise_history`` control
      the Monte-Carlo cost of that field (keep modest for dense ``K``).
    """
    diagonal = cov.ndim == 1
    covbar = cov / jnp.sqrt(cov) if diagonal else cov / jnp.sqrt(jnp.diag(cov))[:, None]
    cov_chol = jnp.sqrt(cov) if diagonal else jnp.linalg.cholesky(cov)
    d, m = params.shape
    iters = int(T / dt)

    def step(carry, i):
        params, key = carry
        B = make_B(params, optimal_params, cov)
        risk = problem.risk_from_B(B)
        key, k_phi, k_noise = jax.random.split(key, 3)
        phi = phi_from_B(B, problem.f, beta1, beta2, k_phi, num_samples=num_samples)
        if diagonal:
            mean = covbar[:, None] * params * phi[:m] + covbar[:, None] * optimal_params * phi[m:]
        else:
            mean = covbar @ params * phi[:m] + covbar @ optimal_params * phi[m:]

        if diagonal:
            k_cov, k_w = jax.random.split(k_noise)
            W = jax.random.normal(k_w, optimal_params.shape) * jnp.sqrt(dt)
            sigma = cov_from_B(B, problem.f, beta1, beta2, k_cov, num_samples=num_samples)
            noise = W @ jnp.linalg.cholesky(sigma)
        else:
            field = adam_noise_field(B, problem.f, beta1, beta2, k_noise, cov_chol,
                                     num_samples=noise_samples,
                                     history_length=noise_history, eps=eps)
            noise = field * jnp.sqrt(dt)

        lr_t = _lr_at(lr, i * dt)
        params = params - lr_t * mean * dt + lr_t * noise / jnp.sqrt(d)
        return (params, key), risk

    (_, _), risks = lax.scan(step, (params, key), jnp.arange(iters))
    return risks, jnp.arange(iters) * dt


def run_sgd_sde(problem: Problem, params, optimal_params, cov, T, lr, *,
                dt=0.005, key):
    sqrtcov = jnp.sqrt(cov) if cov.ndim == 1 else jnp.linalg.cholesky(cov)
    d, m = params.shape
    iters = int(T / dt)

    def step(carry, i):
        params, key = carry
        B = make_B(params, optimal_params, cov)
        risk = problem.risk_from_B(B)
        key, k_h, k_i, k_w = jax.random.split(key, 4)
        W = jax.random.normal(k_w, optimal_params.shape) * jnp.sqrt(dt)
        H = compute_H(B, problem.f, k_h)
        I = compute_I(B, problem.f, k_i)
        vals, vecs = jnp.linalg.eigh(I)
        sqrtI = vecs @ (jnp.sqrt(vals)[:, None] * vecs.T)
        lr_t = _lr_at(lr, i * dt)
        if cov.ndim == 1:
            mean = cov[:, None] * params * H[:m] + cov[:, None] * optimal_params * H[m:]
            params = params - lr_t * mean * dt + lr_t * W @ sqrtI * jnp.sqrt(cov)[:, None] / jnp.sqrt(d)
        else:
            mean = cov @ params * H[:m] + cov @ optimal_params * H[m:]
            params = params - lr_t * mean * dt + lr_t * (sqrtcov @ W @ sqrtI) / jnp.sqrt(d)
        return (params, key), risk

    (_, _), risks = lax.scan(step, (params, key), jnp.arange(iters))
    return risks, jnp.arange(iters) * dt
