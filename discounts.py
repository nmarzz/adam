"""Monte-Carlo functionals of the low-dimensional statistic ``B``.

These compute the drift ``phi`` and diffusion ``sigma`` coefficients of the HAdam
SDE / ODE (and the SGD analogues ``H``, ``I``) by sampling the Gaussian vector
``Q ~ N(0, B)`` and the label noise ``z``. ``f`` is passed as a static argument so
the kernels can be jitted once per problem.

Ported from the old ``risks_and_discounts.py`` with the Python ``for l in range``
loop in :func:`cov_from_B` replaced by a single vectorized expression.
"""
from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp


def _decay(beta, length, scale_by_one_minus=False, start=0):
    powers = beta ** jnp.arange(start, start + length)
    return powers * (1 - beta) if scale_by_one_minus else powers


@partial(jax.jit, static_argnames=["f", "num_samples", "history_length"])
def phi_from_B(B, f, beta1, beta2, key, num_samples=100_000, history_length=500):
    """Drift coefficient ``phi`` of the HAdam SDE."""
    key_Q, key_Q_hist, key_z, key_z_hist = jax.random.split(key, 4)
    Binv = jnp.linalg.pinv(B, hermitian=True)

    Q = jax.random.multivariate_normal(key_Q, jnp.zeros(len(B)), B, shape=(num_samples, 1))
    z = jax.random.normal(key_z, (num_samples, 1))
    Q_hist = jax.random.multivariate_normal(
        key_Q_hist, jnp.zeros(len(B)), B, shape=(num_samples, history_length))
    z_hist = jax.random.normal(key_z_hist, (num_samples, history_length))

    decay2 = _decay(beta2, history_length, start=1)
    decay1 = _decay(beta1, history_length, start=1)

    fq = f(Q).squeeze(axis=1)            # (num_samples, m)
    Q = Q.squeeze(axis=1)                # (num_samples, 2m)

    second_moment_hist = jnp.einsum("nhm,h->nm", f(Q_hist) ** 2 * z_hist[:, :, None] ** 2, decay2)
    second_moment = jnp.sqrt((1 - beta2) * (second_moment_hist + z ** 2 * fq ** 2))

    fmh = f(Q_hist) * z_hist[:, :, None] ** 2 / second_moment[:, None, :]
    fmh = jnp.concatenate([fmh, fmh], axis=-1)
    fmh_w_Q = fmh * (Q_hist @ Binv)

    current = z ** 2 * fq / second_moment
    current = (jnp.concatenate([current, current], axis=-1) * (Q @ Binv)).mean(axis=0)
    history = jnp.einsum("nhm,h->nm", fmh_w_Q, decay1).mean(axis=0)

    return (1 - beta1) * (current + history)


@partial(jax.jit, static_argnames=["f", "num_samples", "history_length"])
def cov_from_B(B, f, beta1, beta2, key, num_samples=100_000, history_length=100):
    """Diffusion (covariance) coefficient ``sigma`` of the HAdam SDE.

    Vectorized form of the old per-``l`` loop: for each history position ``l`` the
    current gradient replaces that slot in the second-moment accumulator. Writing
    ``S`` for the full decayed sum, the ``l``-th accumulator is
    ``S + decay2[l] * (current_grad2 - base[l])``, which broadcasts over ``l``.
    """
    key_Q, key_Q_hist, key_z, key_z_hist = jax.random.split(key, 4)

    Q = jax.random.multivariate_normal(key_Q, jnp.zeros(len(B)), B, shape=(num_samples, 1))
    z = jax.random.normal(key_z, (num_samples, 1))
    Q_hist = jax.random.multivariate_normal(
        key_Q_hist, jnp.zeros(len(B)), B, shape=(num_samples, history_length))
    z_hist = jax.random.normal(key_z_hist, (num_samples, history_length))

    decay1 = _decay(beta1, history_length, scale_by_one_minus=True)
    decay2 = _decay(beta2, history_length, scale_by_one_minus=True)

    fq = f(Q).squeeze(axis=1)            # (num_samples, m)
    current_grad = fq * z                # (num_samples, m)
    current_grad2 = current_grad ** 2

    base = f(Q_hist) ** 2 * z_hist[:, :, None] ** 2          # (n, H, m)
    S = jnp.einsum("h,nhm->nm", decay2, base)                # (n, m): full decayed sum
    # (n, H, m): position-l accumulator with current_grad2 swapped into slot l
    sm = jnp.sqrt(S[:, None, :] + decay2[None, :, None] * (current_grad2[:, None, :] - base))

    contributions = current_grad[:, None, :] / sm            # (n, H, m)
    update = jnp.einsum("h,nhm->nm", decay1, contributions)  # (n, m)

    return jnp.einsum("nm,nk->mk", update, update) / num_samples


@partial(jax.jit, static_argnames=["f", "num_samples", "history_length"])
def adam_U_samples(B, f, beta1, beta2, key, cov_chol, *,
                   num_samples=2000, history_length=50, eps=0.0):
    """Monte-Carlo samples of the field ``U`` underlying the HAdam diffusion.

    Writing (``main.tex`` eq:Sigma)

        ``U^i_k = (1-b1)/sqrt(1-b2) * sum_l b1^l x^i f_k(q) / sqrt(g(q,k,i))``,

    one has ``Sigma_{ijkl} = E[U^i_k U^j_l]``. This returns the raw ``(n, D, M)``
    samples so that both the SDE noise field (:func:`adam_noise_field`) and the ODE
    diffusion (:func:`adam_ode_diffusion`) can be built from them without ever
    materializing the ``D x D`` tensor. ``x ~ N(0, K)`` couples the ambient
    coordinates through ``K``; for diagonal ``K`` this reduces to ``sigma-hat ⊗ Id``.

    ``cov_chol`` is ``sqrt(diag K)`` (shape ``(D,)``) if ``K`` is diagonal, else the
    Cholesky factor with ``K = cov_chol cov_chol^T`` (shape ``(D, D)``).
    """
    diagonal = cov_chol.ndim == 1
    D = cov_chol.shape[0]
    n, H = num_samples, history_length
    k_q, k_qh, k_x, k_xh = jax.random.split(key, 4)

    q = jax.random.multivariate_normal(k_q, jnp.zeros(len(B)), B, shape=(n,))      # (n, M+)
    q_hist = jax.random.multivariate_normal(k_qh, jnp.zeros(len(B)), B, shape=(n, H))

    def draw_x(key, shape):
        g = jax.random.normal(key, shape + (D,))
        return g * cov_chol if diagonal else g @ cov_chol.T

    x = draw_x(k_x, (n,))            # (n, D)
    x_hist = draw_x(k_xh, (n, H))    # (n, H, D)

    fq = f(q)                        # (n, M)
    fqh = f(q_hist)                  # (n, H, M)

    decay1 = beta1 ** jnp.arange(H) * (1 - beta1)
    decay2 = beta2 ** jnp.arange(H) * (1 - beta2)

    base2 = fqh[:, :, None, :] ** 2 * x_hist[:, :, :, None] ** 2   # (n, H, D, M)
    S = jnp.einsum("h,nhik->nik", decay2, base2)                  # (n, D, M)
    cur = fq[:, None, :] * x[:, :, None]                          # (n, D, M)
    cur2 = cur ** 2
    sm = jnp.sqrt(S[:, None] + decay2[None, :, None, None] * (cur2[:, None] - base2) + eps)
    return jnp.einsum("h,nhik->nik", decay1, cur[:, None] / sm)   # (n, D, M)


@partial(jax.jit, static_argnames=["f", "num_samples", "history_length"])
def adam_noise_field(B, f, beta1, beta2, key, cov_chol, *,
                     num_samples=2000, history_length=50, eps=0.0):
    """One draw of the HAdam SDE diffusion noise field, valid for a **general** ``K``.

    A Gaussian field with covariance ``Sigma`` (eq:Sigma) is sampled as
    ``(1/sqrt(n)) sum_s zeta_s U_s`` with ``zeta_s ~ N(0,1)`` — no ``D x D`` tensor is
    materialized. Returns a ``(D, M)`` array. See :func:`adam_U_samples`.
    """
    key, k_zeta = jax.random.split(key)
    U = adam_U_samples(B, f, beta1, beta2, key, cov_chol,
                       num_samples=num_samples, history_length=history_length, eps=eps)
    zeta = jax.random.normal(k_zeta, (num_samples,))
    return jnp.einsum("n,nik->ik", zeta, U) / jnp.sqrt(num_samples)


@partial(jax.jit, static_argnames=["f", "num_samples", "history_length"])
def adam_ode_diffusion(B, f, beta1, beta2, key, cov_chol, KL, R, *,
                       num_samples=2000, history_length=50, eps=0.0):
    """Per-mode diffusion matrices for the Adam ODE under a **general** ``K``.

    The diffusion injected into the ``j``-th spectral mode ``p_j`` of ``B`` is the
    ``M x M`` matrix ``E[(KL[:,j]·U_k)(R[:,j]·U_l)]`` (Ito covariation of
    ``p_j = theta^T K P_j theta`` with ``P_j = L[:,j] R[:,j]^T``). For diagonal ``K``
    (``L = R = Id``, ``KL = K``) this equals ``var_force[j] * sigma-hat``, exactly the
    old simplified term. ``KL = K @ L`` and ``R`` are the right/left eigenvectors of
    ``Kbar``. Returns a ``(D, M, M)`` array (one matrix per mode), excluding the
    ``lr^2 / D`` prefactor (applied by the caller).
    """
    U = adam_U_samples(B, f, beta1, beta2, key, cov_chol,
                       num_samples=num_samples, history_length=history_length, eps=eps)
    alpha = jnp.einsum("dj,ndk->njk", KL, U)     # (n, D, M): KL[:,j] . U_k
    beta = jnp.einsum("dj,ndl->njl", R, U)       # (n, D, M): R[:,j]  . U_l
    return jnp.einsum("njk,njl->jkl", alpha, beta) / num_samples   # (D, M, M)


@partial(jax.jit, static_argnames=["f", "n_samples"])
def compute_I(B, f, key, n_samples=10_000):
    """SGD diffusion: ``E[f(Q) f(Q)^T]``."""
    Q = jax.random.multivariate_normal(key, jnp.zeros(len(B)), B, shape=(n_samples, 1))
    fq = f(Q).squeeze(axis=1)
    return jnp.einsum("nm,nk->mk", fq, fq) / n_samples


@partial(jax.jit, static_argnames=["f", "n_samples"])
def compute_H(B, f, key, n_samples=10_000):
    """SGD drift: ``E[f(Q) (B^{-1} Q)]``."""
    Binv = jnp.linalg.inv(B)
    Q = jax.random.multivariate_normal(key, jnp.zeros(len(B)), B, shape=(n_samples, 1))
    fq = f(Q).squeeze(axis=1)
    Q = Q.squeeze(axis=1)
    fq = jnp.concatenate([fq, fq], axis=1)
    return jnp.mean(fq * (Q @ Binv), axis=0)
