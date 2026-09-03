"""Monte-Carlo functionals of the low-dimensional statistic ``B``.

These compute the drift ``phi`` and diffusion ``Sigma_0`` coefficients of the HAdam
SDE / ODE (and the SGD analogues ``H``, ``I``) by sampling the Gaussian vector
``Q ~ N(0, B)`` and the label noise ``z``. ``f`` is passed as a static argument so
the kernels can be jitted once per problem.

The corrected diffusion is the order-zero second moment from the appendix:
``Sigma_0(B) = E[A_0 \\otimes A_0]``. Its finite-history estimator is represented
by samples of ``A_0`` so neither the SDE nor the ODE needs a fourth-order tensor.
"""
from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp


def _decay(beta, length, scale_by_one_minus=False, start=0):
    powers = beta ** jnp.arange(start, start + length)
    return powers * (1 - beta) if scale_by_one_minus else powers


@partial(jax.jit, static_argnames=["f", "num_samples", "history_length"])
def drift_matrix_from_B(B, f, beta1, beta2, key, num_samples=100_000,
                        history_length=500, eps=0.0, noise_std=0.0):
    """Return the matrix drift ``V(B) = B^+ phi(B)``.

    The shape is ``(M+M*, M)``.  This reduces to the former vector result when
    output coordinates decouple and also supports coupled losses such as
    multiclass softmax. ``noise_std`` adds independent centered Gaussian noise
    to the gradient profile, as required by noisy linear regression.
    """
    key_Q, key_Q_hist, key_z, key_z_hist = jax.random.split(key, 4)
    key_noise = jax.random.fold_in(key, 1)
    key_noise_hist = jax.random.fold_in(key, 2)
    Binv = jnp.linalg.pinv(B, hermitian=True)

    Q = jax.random.multivariate_normal(key_Q, jnp.zeros(len(B)), B, shape=(num_samples, 1))
    z = jax.random.normal(key_z, (num_samples, 1))
    Q_hist = jax.random.multivariate_normal(
        key_Q_hist, jnp.zeros(len(B)), B, shape=(num_samples, history_length))
    z_hist = jax.random.normal(key_z_hist, (num_samples, history_length))

    decay2 = _decay(beta2, history_length, start=1)
    decay1 = _decay(beta1, history_length, start=1)

    fq = f(Q).squeeze(axis=1)            # (num_samples, m)
    fq = fq - noise_std * jax.random.normal(key_noise, fq.shape)
    fqh = f(Q_hist)
    fqh = fqh - noise_std * jax.random.normal(key_noise_hist, fqh.shape)
    Q = Q.squeeze(axis=1)                # (num_samples, 2m)

    second_moment_hist = jnp.einsum("nhm,h->nm", fqh ** 2 * z_hist[:, :, None] ** 2, decay2)
    second_moment = jnp.sqrt((1 - beta2) * (second_moment_hist + z ** 2 * fq ** 2) + eps)

    score_history = Q_hist @ Binv
    scaled_history = fqh * z_hist[:, :, None] ** 2 / second_moment[:, None, :]
    history_outer = score_history[:, :, :, None] * scaled_history[:, :, None, :]
    history = jnp.einsum("h,nhar->nar", decay1, history_outer).mean(axis=0)

    score = Q @ Binv
    scaled_current = z ** 2 * fq / second_moment
    current = (score[:, :, None] * scaled_current[:, None, :]).mean(axis=0)

    return (1 - beta1) * (current + history)


@partial(jax.jit, static_argnames=["f", "num_samples", "history_length"])
def phi_from_B(B, f, beta1, beta2, key, num_samples=100_000,
               history_length=500, eps=0.0, noise_std=0.0):
    """Legacy coordinate-decoupled view of :func:`drift_matrix_from_B`.

    Existing scalar-output callers receive the same length-two vector as
    before. New coupled-output dynamics should consume the full drift matrix.
    """
    V = drift_matrix_from_B(
        B, f, beta1, beta2, key, num_samples=num_samples,
        history_length=history_length, eps=eps, noise_std=noise_std,
    )
    m = V.shape[1]
    return jnp.concatenate([jnp.diag(V[:m]), jnp.diag(V[m:])])


@partial(jax.jit, static_argnames=["f", "num_samples", "history_length"])
def sigma0_samples(B, f, beta1, beta2, key, corr_chol, *,
                   num_samples=2000, history_length=50, eps=0.0,
                   noise_std=0.0):
    """Return finite-history samples of ``A_0`` from appendix eq. Sigma_0.

    ``corr_chol`` factors the normalized coordinate covariance ``C``. For
    diagonal data pass a length-one vector, which samples only one representative
    coordinate and enables the cheap diagonal contraction.
    """
    n, H, D = num_samples, history_length, corr_chol.shape[0]
    key_q, key_x = jax.random.split(key)
    key_noise = jax.random.fold_in(key, 1)
    sequence = 2 * H - 1
    q = jax.random.multivariate_normal(
        key_q, jnp.zeros(B.shape[0]), B, shape=(n, sequence)
    )
    normal = jax.random.normal(key_x, (n, sequence, D))
    xi = normal * corr_chol if corr_chol.ndim == 1 else normal @ corr_chol.T
    fq = f(q)
    fq = fq - noise_std * jax.random.normal(key_noise, fq.shape)
    h = fq[:, :, None, :] ** 2 * xi[:, :, :, None] ** 2

    # Sample H-1 is sample 0.  G_l uses samples l, l-1, ... with the
    # corresponding beta2 weights 1, beta2, ... .  Keeping this orientation is
    # essential: sample 0 must enter G_l with weight beta2**l.
    ell = jnp.arange(H)[:, None]
    lag = jnp.arange(H)[None, :]
    windows = h[:, H - 1 + ell - lag]
    decay2 = beta2 ** jnp.arange(H)
    denom = jnp.sqrt((1 - beta2) * jnp.einsum("h,nlhdm->nldm", decay2, windows) + eps)
    decay1 = (1 - beta1) * beta1 ** jnp.arange(H)
    f0xi0 = fq[:, H - 1, None, :] * xi[:, H - 1, :, None]
    return f0xi0 * jnp.einsum("l,nldm->ndm", decay1, 1.0 / denom)


@partial(jax.jit, static_argnames=["f", "num_samples", "history_length"])
def sigma0_diag(B, f, beta1, beta2, key, *,
                num_samples=2000, history_length=50, eps=0.0,
                noise_std=0.0):
    """Return ``A_0`` samples for one diagonal coordinate, shape ``(n, M)``."""
    return sigma0_samples(
        B, f, beta1, beta2, key, jnp.ones(1),
        num_samples=num_samples, history_length=history_length, eps=eps,
        noise_std=noise_std,
    )[:, 0, :]


@partial(jax.jit, static_argnames=["f", "num_samples", "history_length"])
def cov_from_B(B, f, beta1, beta2, key, num_samples=100_000, history_length=100,
               eps=0.0, noise_std=0.0):
    """Compatibility wrapper returning the diagonal-coordinate ``Sigma_0``."""
    samples = sigma0_diag(B, f, beta1, beta2, key,
                          num_samples=num_samples, history_length=history_length,
                          eps=eps, noise_std=noise_std)
    return jnp.einsum("nm,nk->mk", samples, samples) / num_samples


@partial(jax.jit, static_argnames=["f", "num_samples", "history_length"])
def adam_U_samples(B, f, beta1, beta2, key, cov_chol, *,
                   num_samples=2000, history_length=50, eps=0.0):
    """Legacy samples of the pre-correction diffusion field.

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
                     num_samples=2000, history_length=50, eps=0.0,
                     noise_std=0.0):
    """One draw of the HAdam SDE diffusion noise field, valid for a **general** ``K``.

    A Gaussian field with covariance ``Sigma_0`` is sampled as
    ``(1/sqrt(n)) sum_s zeta_s U_s`` with ``zeta_s ~ N(0,1)`` — no ``D x D`` tensor is
    materialized. Returns a ``(D, M)`` array. ``cov_chol`` factors the normalized
    coordinate covariance ``C`` used by :func:`sigma0_samples`.
    """
    key, k_zeta = jax.random.split(key)
    U = sigma0_samples(B, f, beta1, beta2, key, cov_chol,
                       num_samples=num_samples, history_length=history_length,
                       eps=eps, noise_std=noise_std)
    zeta = jax.random.normal(k_zeta, (num_samples,))
    return jnp.einsum("n,nik->ik", zeta, U) / jnp.sqrt(num_samples)


@partial(jax.jit, static_argnames=["f", "num_samples", "history_length"])
def adam_ode_diffusion(B, f, beta1, beta2, key, cov_chol, KL, R, *,
                       num_samples=2000, history_length=50, eps=0.0,
                       noise_std=0.0):
    """Per-mode diffusion matrices for the Adam ODE under a **general** ``K``.

    The diffusion injected into the ``j``-th spectral mode ``p_j`` of ``B`` is the
    ``M x M`` matrix ``E[(KL[:,j]·U_k)(R[:,j]·U_l)]`` (Ito covariation of
    ``p_j = theta^T K P_j theta`` with ``P_j = L[:,j] R[:,j]^T``). For diagonal ``K``
    (``L = R = Id``, ``KL = K``) this equals ``var_force[j] * sigma-hat``, exactly the
    old simplified term. ``KL = K @ L`` and ``R`` are the right/left eigenvectors of
    ``Kbar``. Returns a ``(D, M, M)`` array (one matrix per mode), excluding the
    ``lr^2 / D`` prefactor (applied by the caller).
    """
    U = sigma0_samples(B, f, beta1, beta2, key, cov_chol,
                       num_samples=num_samples, history_length=history_length,
                       eps=eps, noise_std=noise_std)
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
