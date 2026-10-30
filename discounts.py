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

import math
from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import lax

from utils import sym_sqrt


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


# --------------------------------------------------------------------------- #
# Stationary coefficients from simulated paths (gmm_adam.tex, "Numerical
# evaluation of the coefficients"), with mini-batches and class means.
# --------------------------------------------------------------------------- #
def path_layout(beta1, beta2, window, tol=1e-8):
    """``(burn, window, tail)`` so that ``beta2**burn, beta1**tail < tol``."""
    burn = 0 if beta2 == 0 else int(math.ceil(math.log(tol) / math.log(beta2)))
    tail = 0 if beta1 == 0 else int(math.ceil(math.log(tol) / math.log(beta1)))
    return burn, window, tail


def draw_paths(key, paths, layout, *, batch, inputs, outputs):
    """Common random numbers for :func:`stationary_coefficients`.

    Reusing one draw at every ODE step makes the vector field a smooth
    deterministic function of ``(B, O)``.
    """
    burn, window, tail = layout
    length = burn + window + tail
    k = jax.random.split(key, 5)
    return {
        "z": jax.random.normal(k[0], (paths, length, batch, inputs)),
        "u": jax.random.uniform(k[1], (paths, length, batch)),
        "w": jax.random.normal(k[2], (paths, length, batch, outputs)),
        "xi": jax.random.normal(k[3], (paths, length, batch)),
        "xi2": jax.random.normal(k[4], (paths, length, batch)),
    }


def _recurrence(decay, values, reverse=False):
    """``x_t = decay * x_{t-1} + values_t`` along axis 1 (``x_{t+1}`` if reverse)."""
    def combine(left, right):
        return left[0] * right[0], right[0] * left[1] + right[1]

    decays = jnp.full(values.shape, decay, values.dtype)
    return lax.associative_scan(combine, (decays, values), reverse=reverse, axis=1)[1]


@partial(jax.jit, static_argnames=["f", "burn", "tail", "sgd", "stein"])
def stationary_coefficients(B, O, priors, draws, f, beta1, beta2, eps, *,
                            burn, tail, sgd=False, stein=False):
    """Coefficients of the mini-batch Adam ODE at ``(B, O)``.

    Paths of the stationary Adam recursion on batch-averaged normalized
    gradients ``y_t = b^{-1} sum_beta f(q_{t beta}; c_{t beta}) xi_{t beta}`` give,
    with ``R_t = sum_l beta1^l S_{t+l}^{-1/2}`` and
    ``R'_t = sum_l (beta1 beta2)^l S_{t+l}^{-3/2}`` (every window step is a
    valid "batch 0"),

    * ``phi`` ``(M+, M)``: drift along ``Kbar Theta^+`` (eq. batch_phi),
    * ``chi`` ``(C, M)``: drift along ``D^{-1} Xbar`` (eq. batch_chi),
    * ``sigma`` ``(M, M)``: ``E[A_r A_s]``, the diagonal diffusion ``sigma-hat``,
    * ``tau`` ``(M, M)``: ``sum_beta E_q[E(xi_0beta A_r|q) E(xi_0beta A_s|q)]``, the
      coefficient of the coordinate correlation ``C_ik`` in the off-diagonal
      diffusion, ``Sigma_ik = C_ik tau + O(C_ik^3)``.

    ``stein=True`` estimates ``phi`` from ``B phi = sum_beta E[(q_beta - O_c) xi_beta A]``
    (needs ``B`` nonsingular). The derivative form contains
    ``S^{-1/2} - y^2 S^{-3/2}``, which is close to a delta function at ``y = 0``
    when ``eps`` is tiny and ``beta2`` small, so its Monte-Carlo error blows up
    there; the Stein form has no such term.

    ``sgd=True`` returns the SGD analogues ``E[grad f]``, ``diag(pi) E[f|c]`` and
    ``E[f f^T] / b`` (with ``tau = sigma``).
    """
    z, u, w, xi, xi2 = draws["z"], draws["u"], draws["w"], draws["xi"], draws["xi2"]
    length, batch = xi.shape[1], xi.shape[2]
    window = slice(burn, length - tail)
    classes = jnp.searchsorted(jnp.cumsum(priors)[:-1], u, side="right")
    onehot = jax.nn.one_hot(classes[:, window], len(priors), dtype=z.dtype)
    q = O.T[classes] + z @ sym_sqrt(B)
    profile = jnp.vectorize(f, signature="(p),(),(m)->(m)")(q, classes, w)
    jacobian = jnp.vectorize(jax.jacfwd(f), signature="(p),(),(m)->(m,p)")(
        q[:, window], classes[:, window], w[:, window])          # (n, T, b, M, M+)

    if sgd:
        fw = profile[:, window]
        count = fw.shape[0] * fw.shape[1] * batch
        sigma = jnp.einsum("ntbr,ntbs->rs", fw, fw) / count / batch
        return {
            "phi": jnp.einsum("ntbrs->sr", jacobian) / count,
            "chi": jnp.einsum("ntbc,ntbr->cr", onehot, fw) / count,
            "sigma": sigma,
            "tau": sigma,
        }

    c_beta = (1 - beta1) / jnp.sqrt(1 - beta2)
    floor = eps / (1 - beta2)

    def contributions(normal):
        y = jnp.mean(profile * normal[..., None], axis=2)                    # (n, L, M)
        S = _recurrence(beta2, y ** 2) + floor
        R = _recurrence(beta1, S ** -0.5, reverse=True)
        return y[:, window], S, R[:, window]

    y, S, R = contributions(xi)
    y2, _, R2 = contributions(xi2)
    Rp = _recurrence(beta1 * beta2, S ** -1.5, reverse=True)[:, window]
    xw, xw2 = xi[:, window], xi2[:, window]
    count = y.shape[0] * y.shape[1]

    if stein:
        centered = q[:, window] - O.T[classes[:, window]]
        phi = batch * jnp.linalg.solve(
            B, jnp.einsum("ntbp,ntb,ntr->pr", centered, xw, y * R))
    else:
        phi = jnp.einsum("ntb,ntbrs,ntr->sr", xw ** 2, jacobian, R - y ** 2 * Rp)
    chi = jnp.einsum("ntbc,ntb,ntr->cr", onehot, xw, y * R)
    sigma = jnp.einsum("ntr,nts->rs", y * R, y * R)
    tau = jnp.einsum("ntb,ntr,nts->rs", xw * xw2, y * R, y2 * R2)
    return {
        "phi": c_beta * phi / (batch * count),
        "chi": c_beta * chi / count,
        "sigma": c_beta ** 2 * sigma / count,
        "tau": c_beta ** 2 * (tau + tau.T) / (2 * count),
    }


# --------------------------------------------------------------------------- #
# Softmax mixtures with many rare classes (experiments/zipf_adam_plan.md)
#
# Every coefficient is a sum over the class of one "sample 0",
# ``coef = sum_c pi_c E[contribution | c_{01} = c]``. The class-conditional
# expectations are estimated from pooled events: every window sample of
# ordinary paths (classes ~ pi), plus forced events -- one sample whose class is
# set by hand -- on extra paths for classes too rare to be seen often enough.
# A forced path carries one event per rare class (so a class never contaminates
# its own denominator), and events are spaced so that forced samples are at most
# ``forced_fraction`` of all samples. The softmax Jacobian
# ``d f_r / d q_s = p_r (delta_rs - p_s)`` is contracted on the fly, and paths are
# processed in chunks, so no ``(paths, T, b, C, C)`` tensor is ever formed.
# --------------------------------------------------------------------------- #
class SoftmaxLayout(NamedTuple):
    """Static shapes of a :func:`softmax_coefficients` estimate (hashable)."""
    classes: int
    batch: int
    burn: int
    window: int          # "sample 0" times per ordinary path
    tail: int
    plain_paths: int
    plain_chunk: int     # ordinary paths per chunk
    forced_paths: int
    forced_chunk: int
    events: int          # forced events per forced path (= number of rare classes)
    spacing: int         # steps between forced events


class SoftmaxPlan(NamedTuple):
    """Arrays of a :func:`softmax_coefficients` estimate (common random numbers)."""
    key_plain: jnp.ndarray
    key_forced: jnp.ndarray
    forced: jnp.ndarray   # (forced_paths, events) class of each forced event
    weights: jnp.ndarray  # (C,) pi_c / pooled number of class-c events
    counts: jnp.ndarray   # (2, C) ordinary and forced class-c events


def _softmax_chunk_draws(key, index, paths, length, batch, classes, cdf):
    k_u, k_z, k_x = jax.random.split(jax.random.fold_in(key, index), 3)
    u = jax.random.uniform(k_u, (paths, length, batch))
    labels = jnp.searchsorted(cdf, u, side="right")
    z = jax.random.normal(k_z, (paths, length, batch, classes))
    xi = jax.random.normal(k_x, (paths, length, batch))
    return labels, z, xi


def softmax_plan(key, priors, beta1, beta2, *, batch=1, plain_batches=2 ** 18,
                 window=1024, min_events=1000, forced_fraction=0.02,
                 chunk_elements=2 ** 24, min_chunks=1, tol=1e-8):
    """Layout and common random numbers for :func:`softmax_coefficients`.

    ``plain_batches`` ordinary batches (paths x window) are used as "sample 0";
    every class with fewer than ``min_events`` ordinary events gets
    ``min_events`` forced events. ``min_chunks > 1`` splits both kinds of paths
    into at least that many chunks (for batch-means standard errors).
    """
    priors = jnp.asarray(priors)
    C = len(priors)
    burn, window, tail = path_layout(beta1, beta2, window, tol)
    plain_length = burn + window + tail
    plain_chunk = max(1, min(plain_batches // window // min_chunks,
                             chunk_elements // (plain_length * batch * C)))
    plain_paths = plain_chunk * max(1, -(-(plain_batches // window) // plain_chunk))
    k_plain, k_forced, k_perm = jax.random.split(key, 3)
    cdf = jnp.cumsum(priors)[:-1]

    @jax.jit
    def chunk_counts(i):
        u = jax.random.uniform(jax.random.split(jax.random.fold_in(k_plain, i), 3)[0],
                               (plain_chunk, plain_length, batch))
        labels = jnp.searchsorted(cdf, u[:, burn:burn + window], side="right")
        return jnp.bincount(labels.ravel(), length=C)

    plain = sum(chunk_counts(i) for i in range(plain_paths // plain_chunk))
    rare = jnp.flatnonzero(plain < min_events)
    events = int(len(rare))
    spacing = max(1, math.ceil(1 / (forced_fraction * batch)))
    forced_paths = min_events if events else 0
    forced_length = burn + max(events - 1, 0) * spacing + 1 + tail
    forced_chunk = max(1, min(max(forced_paths // min_chunks, 1),
                              chunk_elements // (forced_length * batch * C)))
    forced_paths = forced_chunk * -(-forced_paths // forced_chunk) if events else 0
    keys = jax.random.split(k_perm, max(forced_paths, 1))
    forced = jax.vmap(lambda k: jax.random.permutation(k, rare))(keys) if events \
        else jnp.zeros((1, 1), int)
    forced_counts = jnp.zeros(C, int).at[rare].set(forced_paths)
    total = plain + forced_counts
    layout = SoftmaxLayout(C, batch, burn, window, tail, plain_paths, plain_chunk,
                           forced_paths, forced_chunk, max(events, 1), spacing)
    plan = SoftmaxPlan(k_plain, k_forced, forced,
                       priors / jnp.maximum(total, 1), jnp.stack([plain, forced_counts]))
    return layout, plan


@partial(jax.jit, static_argnames=["layout", "mode", "stein", "diagnostics"])
def softmax_coefficients(B, O, priors, plan: SoftmaxPlan, beta1, beta2, eps, *,
                         layout: SoftmaxLayout, mode="adam", stein=False,
                         diagnostics=False):
    """``phi``, ``chi``, ``sigma`` (and ``tau = sigma``) for ``C``-class softmax
    regression on a mixture with ``M = C`` logits and no teacher.

    ``mode="adam"`` gives the mini-batch Adam coefficients of
    :func:`stationary_coefficients` (derivative or ``stein`` form of ``phi``);
    ``"sgd"`` gives ``E[grad f]``, ``diag(pi) E[f|c]``, ``E[f f^T]/b``; ``"psgd"``
    is the ``beta2 -> 1`` limit, SGD coefficients with the deterministic
    preconditioner ``(E f_r^2 / b + eps)^{-1/2}`` on output ``r``.
    ``tau`` is not estimated: it does not enter for ``K = Id``.

    With ``diagnostics=True`` also returns per-chunk contributions to the
    diagonals of ``chi``, ``phi`` and ``sigma`` (for batch-means standard errors).
    """
    C, b = layout.classes, layout.batch
    root = sym_sqrt(B)
    cdf = jnp.cumsum(priors)[:-1]
    c_beta = (1 - beta1) / jnp.sqrt(1 - beta2)
    floor = eps / (1 - beta2)
    weights = plan.weights

    def accumulate(labels, noise, p, f, xi, y, R, Rp):
        """Weighted sums over events ``(n, T, b_e)``; ``y, R, Rp`` are batch level."""
        w = weights[labels]
        if mode != "adam":
            out = {
                "phi": jnp.diag(jnp.einsum("ntb,ntbr->r", w, p))
                - jnp.einsum("ntbs,ntbr->sr", w[..., None] * p, p),
                "chi": jax.ops.segment_sum((w[..., None] * f).reshape(-1, C),
                                           labels.ravel(), C),
                "second": jnp.einsum("ntbr,ntbs->rs", w[..., None] * f, f),
            }
            return out
        A = c_beta * y * R
        if stein:
            phi = jnp.einsum("ntbs,ntr->sr", (w * b * xi)[..., None] * noise, A)
        else:
            weight = (w * c_beta * xi ** 2)[..., None] * p
            W = R - y ** 2 * Rp
            phi = (jnp.diag(jnp.einsum("ntbr,ntr->r", weight, W))
                   - jnp.einsum("ntbs,ntbr->sr", weight, p * W[:, :, None, :]))
        values = (w * b * xi)[..., None] * A[:, :, None, :]
        chi = jax.ops.segment_sum(values.reshape(-1, C), labels.ravel(), C)
        sigma = jnp.einsum("ntbr,nts->rs",
                           (w * c_beta * xi)[..., None] * f * R[:, :, None, :], A)
        return {"phi": phi, "chi": chi, "sigma": sigma}

    def chunk(key, index, paths, length, forced_rows=None, times=None):
        labels, z, xi = _softmax_chunk_draws(key, index, paths, length, b, C, cdf)
        if forced_rows is not None:
            labels = labels.at[:, times, 0].set(forced_rows.astype(labels.dtype))
        noise = z @ root
        p = jax.nn.softmax(O.T[labels] + noise, axis=-1)
        f = p - jax.nn.one_hot(labels, C, dtype=p.dtype)
        y = R = Rp = None
        if mode == "adam":
            y = jnp.mean(f * xi[..., None], axis=2)
            S = _recurrence(beta2, y ** 2) + floor
            R = _recurrence(beta1, S ** -0.5, reverse=True)
            Rp = None if stein else _recurrence(beta1 * beta2, S ** -1.5, reverse=True)
        if forced_rows is None:
            window = slice(layout.burn, layout.burn + layout.window)
            pick = lambda a: None if a is None else a[:, window]
            return accumulate(labels[:, window], noise[:, window], p[:, window],
                              f[:, window], xi[:, window], pick(y), pick(R), pick(Rp))
        pick = lambda a: None if a is None else a[:, times]
        first = lambda a: a[:, times, :1]
        return accumulate(first(labels), first(noise), first(p), first(f), first(xi),
                          pick(y), pick(R), pick(Rp))

    def diag_of(part):
        if mode != "adam":
            return jnp.stack([jnp.diag(part["chi"]), jnp.diag(part["phi"]),
                              jnp.diag(part["second"])])
        return jnp.stack([jnp.diag(part["chi"]), jnp.diag(part["phi"]),
                          jnp.diag(part["sigma"])])

    def run(n_chunks, body):
        def step(total, i):
            part = body(i)
            return jax.tree.map(jnp.add, total, part), diag_of(part)
        first = body(0)
        rest, diags = lax.scan(step, first, jnp.arange(1, n_chunks))
        return rest, jnp.concatenate([diag_of(first)[None], diags])

    plain_length = layout.burn + layout.window + layout.tail
    total, plain_diag = run(
        layout.plain_paths // layout.plain_chunk,
        lambda i: chunk(plan.key_plain, i, layout.plain_chunk, plain_length))
    forced_diag = jnp.zeros((0, 3, C))
    if layout.forced_paths:
        times = layout.burn + layout.spacing * jnp.arange(layout.events)
        forced_length = (layout.burn + (layout.events - 1) * layout.spacing + 1
                         + layout.tail)

        def forced_body(i):
            rows = lax.dynamic_slice_in_dim(plan.forced, i * layout.forced_chunk,
                                            layout.forced_chunk)
            return chunk(plan.key_forced, i, layout.forced_chunk, forced_length,
                         rows, times)

        extra, forced_diag = run(layout.forced_paths // layout.forced_chunk, forced_body)
        total = jax.tree.map(jnp.add, total, extra)

    if mode == "adam":
        phi = jnp.linalg.solve(B, total["phi"]) if stein else total["phi"]
        sigma = (total["sigma"] + total["sigma"].T) / 2
        out = {"phi": phi, "chi": total["chi"], "sigma": sigma, "tau": sigma}
    else:
        second = total["second"]
        scale = jnp.ones(C) if mode == "sgd" else lax.rsqrt(jnp.diag(second) / b + eps)
        sigma = second / b * scale[:, None] * scale[None, :]
        out = {"phi": total["phi"] * scale[None, :], "chi": total["chi"] * scale[None, :],
               "sigma": sigma, "tau": sigma}
    if diagnostics:
        out["plain_diag"], out["forced_diag"] = plain_diag, forced_diag
    return out


def softmax_standard_errors(B, O, priors, plan: SoftmaxPlan, beta1, beta2, eps, *,
                            layout: SoftmaxLayout, mode="adam", stein=False):
    """Diagonals of ``chi``, ``phi``, ``sigma`` with batch-means standard errors.

    Chunks of ordinary paths are i.i.d., and so are chunks of forced paths; the
    estimate is the sum of all chunk contributions, so its variance is
    ``n_p var(plain chunk) + n_f var(forced chunk)``. Use a plan built with
    ``min_chunks >= 16``. Returns ``(estimate, standard_error)``, each ``(3, C)``.
    """
    out = softmax_coefficients(B, O, priors, plan, beta1, beta2, eps, layout=layout,
                               mode=mode, stein=stein, diagnostics=True)
    estimate = out["plain_diag"].sum(0) + out["forced_diag"].sum(0)
    var = sum(len(d) * jnp.var(d, axis=0, ddof=1) for d in (out["plain_diag"], out["forced_diag"])
              if len(d) > 1)
    return estimate, jnp.sqrt(var)


@partial(jax.jit, static_argnames=["mode"])
def softmax_moment_coefficients(B, O, priors, draws, eps, batch, *, mode="sgd"):
    """SGD (``mode="sgd"``) or ``beta2 -> 1`` PSGD (``"psgd"``) coefficients for
    softmax on a mixture, from exactly class-stratified expectations.

    These coefficients are plain expectations over one sample, so no Adam paths
    are needed: with fixed draws ``g`` (``(n, C)``, common random numbers) and
    ``q | c = O[:, c] + sym_sqrt(B) g``, every class gets all ``n`` draws and is
    weighted by ``pi_c`` -- rare classes are as accurate as frequent ones.
    Returns the same dictionary as :func:`softmax_coefficients`.
    """
    C = len(priors)
    noise = draws @ sym_sqrt(B)
    n = noise.shape[0]

    def one_class(acc, c):
        p = jax.nn.softmax(O[:, c][None, :] + noise, axis=-1)
        f = p - jax.nn.one_hot(c, C, dtype=p.dtype)
        mean_p, mean_f = p.mean(0), f.mean(0)
        acc = (acc[0] + priors[c] * mean_p, acc[1] + priors[c] * (p.T @ p) / n,
               acc[2] + priors[c] * (f.T @ f) / n)
        return acc, priors[c] * mean_f

    zero = jnp.zeros((C, C), noise.dtype)
    (mean_p, outer_p, second), chi = lax.scan(one_class, (jnp.zeros(C, noise.dtype), zero, zero),
                                              jnp.arange(C))
    phi = jnp.diag(mean_p) - outer_p
    scale = jnp.ones(C) if mode == "sgd" else lax.rsqrt(jnp.diag(second) / batch + eps)
    sigma = second / batch * scale[:, None] * scale[None, :]
    return {"phi": phi * scale[None, :], "chi": chi * scale[None, :], "sigma": sigma,
            "tau": sigma}
