"""Gaussian-mixture extension of the high-dimensional Adam ODEs.

Notation follows ``paper/gmm_adam.tex``; code names in brackets.

Data model
----------
A class ``c ~ Categorical(probs)`` is drawn and then ``x | c ~ N(xbar_c, K)``;
the covariance ``K`` is shared by all components and the class means are the
columns of Xbar [``means``, shape ``(D, C)``] with ``||xbar_c|| = O(1)``.  The
model is a linear composite ``L(x, c; Theta) = F(Theta^T x; c)`` with
``Theta in R^{D x M}``, so the per-sample gradient is ``x f(Theta^T x; c)^T``.

Order parameters
----------------
``B = Theta^T K Theta`` (``M x M``) and the mean overlaps ``O = Theta^T Xbar``
[``Mbar``, ``M x C``].  Given the class, the logits are ``q ~ N(O[:, c], B)``, so
the risk is ``h(B, O)``.

Limiting coefficients
---------------------
Let ``(c_t, q_t, xi_t)_{t in Z}`` be i.i.d. with ``c_t ~ probs``,
``q_t | c_t ~ N(O[:, c_t], B)`` and ``xi_t ~ N(0, 1)``; put
``S_{t,r} = sum_{m>=0} beta2^m f_r(q_{t-m})^2 xi_{t-m}^2 + eps / (1 - beta2)`` and
``A_r = c_beta f_r(q_0) xi_0 sum_{l>=0} beta1^l S_{l,r}^{-1/2}`` (the total
contribution of sample 0 to all later updates).  Then

* ``phi[s, r]   = E[ d/dq_0^s (xi_0 A_r) ]``      drift through ``Kbar Theta``
* ``chi[c, r]   = E[ 1{c_0 = c} xi_0 A_r ]``      new drift through ``Dinv Xbar``
* ``sigma[r, s] = E[ A_r A_s ]``                  diffusion (diagonal ``K``)

with ``Dinv = diag(K)^{-1/2}`` and ``Kbar = Dinv K``.  The HAdam drift is
``-eta (Kbar Theta phi + Dinv Xbar chi)``.  For diagonal ``K`` (mode ``j`` =
coordinate ``j``, ``lam_j = sqrt(K_jj)``, ``xbar^j`` = row ``j`` of Xbar) the ODEs
for ``p_j`` [``P``] and ``o_j`` [``V``] are

    dp_j/dt = -eta lam_j Sym(p_j phi + o_j chi) + eta^2 (K_jj / D) sigma
    do_j/dt = -eta (lam_j phi^T o_j + chi^T xbar^j xbar^j^T / lam_j)

with ``B = sum_j p_j`` and ``O = sum_j o_j``.

The expectations are estimated with long simulated sample paths: one forward
recursion builds every denominator ``S_t`` and one backward recursion builds the
discounted future sums, so each time index of each path is a valid "sample 0".
"""
from __future__ import annotations

from functools import partial
from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from numpy.polynomial.hermite import hermgauss


# --------------------------------------------------------------------------- #
# Problems
# --------------------------------------------------------------------------- #
class GMMProblem(NamedTuple):
    name: str
    num_classes: int      # C, number of mixture components (= labels)
    num_outputs: int      # M, number of trainable output columns of Theta
    f: Callable           # (q (..., M), c (...)) -> dF/dq, shape (..., M)
    jac: Callable         # (q, c) -> (..., M, M), entry [r, s] = d f_r / d q_s
    risk: Callable        # (B, Mbar, probs) -> population risk h(B, Mbar)
    error: Callable       # (B, Mbar, probs) -> misclassification probability


def psd_sqrt(B):
    """Symmetric square root of a PSD matrix (valid when ``B`` is singular)."""
    B = 0.5 * (B + B.T)
    w, V = jnp.linalg.eigh(B)
    return (V * jnp.sqrt(jnp.clip(w, 0.0))) @ V.T


def _gauss_hermite(n):
    x, w = hermgauss(n)
    return jnp.asarray(np.sqrt(2.0) * x), jnp.asarray(w / np.sqrt(np.pi))


def binary_logistic_problem(gh_nodes: int = 80) -> GMMProblem:
    """Two components, label ``y = 2c - 1``, loss ``log(1 + exp(-y q))``, ``M = 1``.

    With ``c in {0, 1}`` the gradient profile is ``f(q; c) = sigmoid(q) - c``.
    """
    nodes, weights = _gauss_hermite(gh_nodes)

    def f(q, c):
        return jax.nn.sigmoid(q) - c[..., None].astype(q.dtype)

    def jac(q, c):
        s = jax.nn.sigmoid(q)
        return (s * (1.0 - s))[..., None]

    def risk(B, Mbar, probs):
        std = jnp.sqrt(jnp.clip(B[0, 0], 0.0))
        y = jnp.array([-1.0, 1.0])
        q = Mbar[0][:, None] + std * nodes[None, :]               # (2, n)
        per_class = jnp.sum(weights * jax.nn.softplus(-y[:, None] * q), axis=1)
        return jnp.sum(probs * per_class)

    def error(B, Mbar, probs):
        std = jnp.sqrt(jnp.clip(B[0, 0], 1e-300))
        y = jnp.array([-1.0, 1.0])
        return jnp.sum(probs * jax.scipy.stats.norm.cdf(-y * Mbar[0] / std))

    return GMMProblem("binary_logistic", 2, 1, f, jac, risk, error)


def softmax_problem(num_classes: int, *, gh_nodes: int = 16,
                    error_samples: int = 200_000, seed: int = 0) -> GMMProblem:
    """``C``-class softmax regression on the component label, full ``M = C`` logits.

    Loss ``F(q; c) = logsumexp(q) - q_c``, profile ``f = softmax(q) - e_c``.  The
    risk uses a tensor Gauss--Hermite rule (``C <= 3``) or fixed Monte Carlo draws.
    """
    C = num_classes
    if C <= 3:
        x, w = _gauss_hermite(gh_nodes)
        grids = jnp.meshgrid(*([x] * C), indexing="ij")
        nodes = jnp.stack([g.ravel() for g in grids], axis=1)            # (n^C, C)
        wgrids = jnp.meshgrid(*([w] * C), indexing="ij")
        weights = jnp.prod(jnp.stack([g.ravel() for g in wgrids], 1), 1)
    else:
        z = jax.random.normal(jax.random.PRNGKey(seed), (error_samples // 2, C))
        nodes = jnp.concatenate([z, -z])
        weights = jnp.full(nodes.shape[0], 1.0 / nodes.shape[0])
    err_z = jax.random.normal(jax.random.PRNGKey(seed + 1), (error_samples, C))

    def f(q, c):
        return jax.nn.softmax(q, axis=-1) - jax.nn.one_hot(c, C, dtype=q.dtype)

    def jac(q, c):
        p = jax.nn.softmax(q, axis=-1)
        return jnp.einsum("...r,rs->...rs", p, jnp.eye(C)) - p[..., :, None] * p[..., None, :]

    def risk(B, Mbar, probs):
        root = psd_sqrt(B)
        q = Mbar.T[:, None, :] + (nodes @ root.T)[None]                 # (C, n, C)
        lse = jax.scipy.special.logsumexp(q, axis=-1) @ weights          # (C,)
        return jnp.sum(probs * (lse - jnp.diag(Mbar)))

    def error(B, Mbar, probs):
        root = psd_sqrt(B)
        q = Mbar.T[:, None, :] + (err_z @ root.T)[None]
        wrong = jnp.argmax(q, axis=-1) != jnp.arange(C)[:, None]
        return jnp.sum(probs * wrong.mean(axis=1))

    return GMMProblem(f"softmax_{C}", C, C, f, jac, risk, error)


# --------------------------------------------------------------------------- #
# Limiting Adam coefficients phi, chi, sigma
# --------------------------------------------------------------------------- #
class CoefficientConfig(NamedTuple):
    num_paths: int = 128      # independent sample paths
    burn_in: int = 400        # warm-up so S_t sees its whole past (beta2^burn_in ~ 0)
    num_used: int = 2048      # time indices per path used as "sample 0"
    tail: int = 200           # look-ahead so the beta1-discounted future sum converges


def coefficient_config(beta1, beta2, *, num_paths=128, num_used=2048,
                       tol=1e-8) -> CoefficientConfig:
    """Burn-in/look-ahead lengths so that ``beta2^burn_in, beta1^tail < tol``."""
    burn = int(np.ceil(np.log(tol) / np.log(beta2))) if beta2 > 0 else 1
    tail = int(np.ceil(np.log(tol) / np.log(beta1))) if beta1 > 0 else 1
    return CoefficientConfig(num_paths, max(burn, 1), num_used, max(tail, 1))


class BaseNoise(NamedTuple):
    """Standardized randomness behind one coefficient estimate.

    Reusing the same draws at every ODE step (common random numbers) makes the
    estimated coefficients smooth deterministic functions of ``(B, Mbar)``.
    """
    u: jnp.ndarray    # (L, n) uniforms -> class labels
    z: jnp.ndarray    # (L, n, M) standard normals -> q = Mbar[:, c] + B^{1/2} z
    xi: jnp.ndarray   # (L, n) standard normals -> normalized data coordinate


def draw_base_noise(key, num_outputs: int,
                    config: CoefficientConfig = CoefficientConfig(),
                    dtype=jnp.float64) -> BaseNoise:
    L = config.burn_in + config.num_used + config.tail
    k_u, k_z, k_x = jax.random.split(key, 3)
    return BaseNoise(jax.random.uniform(k_u, (L, config.num_paths), dtype),
                     jax.random.normal(k_z, (L, config.num_paths, num_outputs), dtype),
                     jax.random.normal(k_x, (L, config.num_paths), dtype))


def adam_coefficients(B, Mbar, probs, key, *, problem: GMMProblem, beta1, beta2,
                      eps, config: CoefficientConfig = CoefficientConfig()):
    """Monte Carlo estimates of ``(phi (M, M), chi (C, M), sigma (M, M))``.

    ``phi[s, r]`` multiplies ``Kbar Theta`` (row ``s`` indexes logits, column ``r``
    outputs), ``chi[c, r]`` multiplies ``Dinv xbar_c`` and ``sigma`` is the
    one-coordinate diffusion matrix for diagonal ``K``.
    """
    noise = draw_base_noise(key, B.shape[0], config, B.dtype)
    return adam_coefficients_from_noise(B, Mbar, probs, noise, problem=problem,
                                        beta1=beta1, beta2=beta2, eps=eps,
                                        config=config)


@partial(jax.jit, static_argnames=["problem", "config"])
def adam_coefficients_from_noise(B, Mbar, probs, noise: BaseNoise, *,
                                 problem: GMMProblem, beta1, beta2, eps,
                                 config: CoefficientConfig = CoefficientConfig()):
    """Same as :func:`adam_coefficients` with the randomness supplied."""
    n, H0, T, H1 = config.num_paths, config.burn_in, config.num_used, config.tail
    M, C = B.shape[0], probs.shape[0]
    dtype = B.dtype
    c = jnp.minimum(jnp.searchsorted(jnp.cumsum(probs), noise.u), C - 1)
    xi = noise.xi
    q = Mbar.T[c] + noise.z @ psd_sqrt(B).T                              # (L, n, M)
    fq = problem.f(q, c)                                                 # (L, n, M)
    xi2 = xi[..., None] ** 2
    h = fq ** 2 * xi2
    delta = eps / (1.0 - beta2)

    def forward(S, h_t):
        S = beta2 * S + h_t
        return S, S

    _, S = lax.scan(forward, jnp.zeros((n, M), dtype), h)
    S = S + delta
    inv_half = lax.rsqrt(S)
    inv_three_half = inv_half / S

    def backward(carry, inputs):
        R1, R3 = carry
        a, b = inputs
        R1 = a + beta1 * R1
        R3 = b + beta1 * beta2 * R3
        return (R1, R3), (R1, R3)

    zeros = jnp.zeros((n, M), dtype)
    _, (R1, R3) = lax.scan(backward, (zeros, zeros), (inv_half, inv_three_half),
                           reverse=True)

    sl = slice(H0, H0 + T)
    cb = (1.0 - beta1) / jnp.sqrt(1.0 - beta2)
    f_u, xi_u, xi2_u, R1_u, R3_u = fq[sl], xi[sl], xi2[sl], R1[sl], R3[sl]
    count = T * n

    A = cb * f_u * xi_u[..., None] * R1_u                               # (T, n, M)
    sigma = jnp.einsum("tnr,tns->rs", A, A) / count

    weight = cb * xi2_u * (R1_u - f_u ** 2 * xi2_u * R3_u)              # (T, n, M)
    J = problem.jac(q[sl], c[sl])                                        # (T, n, M, M)
    phi = jnp.einsum("tnrs,tnr->sr", J, weight) / count

    onehot = jax.nn.one_hot(c[sl], C, dtype=dtype)
    chi = cb * jnp.einsum("tnc,tnr->cr", onehot, f_u * xi2_u * R1_u) / count
    return phi, chi, sigma


def psgd_coefficients(B, Mbar, probs, noise: BaseNoise, *, problem: GMMProblem,
                      eps, config: CoefficientConfig = CoefficientConfig()):
    """``beta2 -> 1`` limit: Adam replaced by momentum SGD with the deterministic
    preconditioner ``1 / sqrt(E v + eps)``.  Used as a baseline only.

        phi[s, r]   = E[d_s f_r] / sqrt(E f_r^2 + eps)
        chi[c, r]   = E[1{c} f_r] / sqrt(E f_r^2 + eps)
        sigma[r, s] = E[f_r f_s] / sqrt((E f_r^2 + eps)(E f_s^2 + eps))
    """
    C = probs.shape[0]
    c = jnp.minimum(jnp.searchsorted(jnp.cumsum(probs), noise.u), C - 1)
    q = Mbar.T[c] + noise.z @ psd_sqrt(B).T
    fq = problem.f(q, c)
    count = fq.shape[0] * fq.shape[1]
    scale = lax.rsqrt(jnp.mean(fq ** 2, axis=(0, 1)) + eps)                # (M,)
    phi = jnp.einsum("lnrs->sr", problem.jac(q, c)) / count * scale[None, :]
    onehot = jax.nn.one_hot(c, C, dtype=B.dtype)
    chi = jnp.einsum("lnc,lnr->cr", onehot, fq) / count * scale[None, :]
    sigma = jnp.einsum("lnr,lns->rs", fq, fq) / count * scale[:, None] * scale[None, :]
    return phi, chi, sigma


# --------------------------------------------------------------------------- #
# ODE (diagonal covariance; mode j = coordinate j)
# --------------------------------------------------------------------------- #
class ODEResult(NamedTuple):
    times: np.ndarray
    B: np.ndarray        # (T, M, M)
    Mbar: np.ndarray     # (T, M, C)
    risk: np.ndarray
    error: np.ndarray


def ode_initial_state(theta0, cov_diag, means):
    P = cov_diag[:, None, None] * theta0[:, :, None] * theta0[:, None, :]  # (D, M, M)
    V = theta0[:, :, None] * means[:, None, :]                             # (D, M, C)
    return P, V


def ode_rhs(P, V, coeffs, lr, cov_diag, means):
    phi, chi, sigma = coeffs
    D = cov_diag.shape[0]
    lam = jnp.sqrt(cov_diag)
    X = jnp.einsum("jas,sr->jar", P, phi) + jnp.einsum("jac,cr->jar", V, chi)
    dP = (-lr * lam[:, None, None] * (X + jnp.swapaxes(X, 1, 2))
          + lr ** 2 * (cov_diag / D)[:, None, None] * sigma)
    mean_force = jnp.einsum("cr,jc->jr", chi, means) / lam[:, None]       # (D, M)
    dV = -lr * (lam[:, None, None] * jnp.einsum("sr,jsc->jrc", phi, V)
                + mean_force[:, :, None] * means[:, None, :])
    return dP, dV


VARIANTS = ("adam", "no_mean", "no_diffusion", "psgd")


def _make_coefficient_fn(problem, probs, beta1, beta2, eps, config, fresh_noise,
                         variant="adam"):
    """Return ``coef(B, Mbar, step_key, fixed_noise) -> (phi, chi, sigma)``.

    With ``fresh_noise=False`` (default) the same base draws ``fixed_noise`` are
    reused at every call (common random numbers), so the vector field is a smooth
    deterministic function; otherwise new draws are made from ``step_key``.

    ``variant`` selects the full Adam coefficients (``"adam"``) or an ablation:
    ``"no_mean"`` drops ``chi``, ``"no_diffusion"`` drops ``sigma`` and ``"psgd"``
    uses the ``beta2 -> 1`` preconditioned-SGD coefficients.
    """
    if variant not in VARIANTS:
        raise ValueError(f"variant must be one of {VARIANTS}")

    def coef(B, Mbar, step_key, fixed_noise):
        noise = (draw_base_noise(step_key, problem.num_outputs, config, B.dtype)
                 if fresh_noise else fixed_noise)
        if variant == "psgd":
            return psgd_coefficients(B, Mbar, probs, noise, problem=problem,
                                     eps=eps, config=config)
        phi, chi, sigma = adam_coefficients_from_noise(
            B, Mbar, probs, noise, problem=problem, beta1=beta1, beta2=beta2,
            eps=eps, config=config)
        if variant == "no_mean":
            chi = jnp.zeros_like(chi)
        if variant == "no_diffusion":
            sigma = jnp.zeros_like(sigma)
        return phi, chi, sigma
    return coef


def run_gmm_ode(problem: GMMProblem, theta0, cov_diag, means, probs, *, lr,
                beta1, beta2, eps, T, dt, key, record_every=1, fresh_noise=False,
                variant="adam", config: CoefficientConfig = CoefficientConfig()):
    """Integrate the per-mode ODE with Heun's method.

    ``lr`` may be a float or a callable of the rescaled time ``t``.  Observables
    are recorded every ``record_every`` steps and at the final time.
    """
    P, V = ode_initial_state(theta0, cov_diag, means)
    steps = int(round(T / dt))
    lr_at = lr if callable(lr) else (lambda t: lr)
    coef = _make_coefficient_fn(problem, probs, beta1, beta2, eps, config,
                                fresh_noise, variant)
    num_records = steps // record_every

    @jax.jit
    def integrate(P, V, key, fixed):
        def step(carry, i):
            P, V, key = carry
            key, k1, k2 = jax.random.split(key, 3)
            t = i * dt
            B, Mbar = P.sum(0), V.sum(0)
            dP1, dV1 = ode_rhs(P, V, coef(B, Mbar, k1, fixed), lr_at(t),
                               cov_diag, means)
            P2, V2 = P + dt * dP1, V + dt * dV1
            coeffs2 = coef(P2.sum(0), V2.sum(0), k2, fixed)
            dP2, dV2 = ode_rhs(P2, V2, coeffs2, lr_at(t + dt), cov_diag, means)
            P = P + 0.5 * dt * (dP1 + dP2)
            V = V + 0.5 * dt * (dV1 + dV2)
            return (P, V, key), (B, Mbar)

        def chunk(carry, start):
            carry, (Bs, Ms) = lax.scan(step, carry, start + jnp.arange(record_every))
            return carry, (Bs[0], Ms[0])

        return lax.scan(chunk, (P, V, key), jnp.arange(num_records) * record_every)

    fixed = draw_base_noise(key, problem.num_outputs, config, P.dtype)
    carry, (B_path, M_path) = integrate(P, V, jax.random.fold_in(key, 1), fixed)
    P, V, _ = carry
    B_path = jnp.concatenate([B_path, P.sum(0)[None]])
    M_path = jnp.concatenate([M_path, V.sum(0)[None]])
    times = np.arange(num_records + 1) * record_every * dt
    risk = jax.vmap(lambda b, m: problem.risk(b, m, probs))(B_path, M_path)
    err = jax.vmap(lambda b, m: problem.error(b, m, probs))(B_path, M_path)
    return ODEResult(times, np.asarray(B_path), np.asarray(M_path),
                     np.asarray(risk), np.asarray(err))


def run_isotropic_ode(problem: GMMProblem, B0, Mbar0, gram, probs, *, lr, beta1,
                      beta2, eps, T, dt, key, fresh_noise=False,
                      config: CoefficientConfig = CoefficientConfig()):
    """Aggregated ODE for ``K = I`` (closed in ``B`` and ``O``; ``gram = Xbar^T Xbar``).

        dB/dt = -eta Sym(B phi + O chi) + eta^2 sigma
        dO/dt = -eta (phi^T O + chi^T gram)

    Used to cross-check :func:`run_gmm_ode`.
    """
    steps = int(round(T / dt))
    coef = _make_coefficient_fn(problem, probs, beta1, beta2, eps, config,
                                fresh_noise)

    def rhs(B, Mb, coeffs):
        phi, chi, sigma = coeffs
        X = B @ phi + Mb @ chi
        return -lr * (X + X.T) + lr ** 2 * sigma, -lr * (phi.T @ Mb + chi.T @ gram)

    @jax.jit
    def integrate(carry, fixed):
        def step(carry, i):
            B, Mb, key = carry
            key, k1, k2 = jax.random.split(key, 3)
            dB1, dM1 = rhs(B, Mb, coef(B, Mb, k1, fixed))
            B2, M2 = B + dt * dB1, Mb + dt * dM1
            dB2, dM2 = rhs(B2, M2, coef(B2, M2, k2, fixed))
            out = (B, Mb)
            return (B + 0.5 * dt * (dB1 + dB2), Mb + 0.5 * dt * (dM1 + dM2), key), out
        return lax.scan(step, carry, jnp.arange(steps))

    fixed = draw_base_noise(key, problem.num_outputs, config, B0.dtype)
    (B, Mb, _), (B_path, M_path) = integrate((B0, Mbar0, jax.random.fold_in(key, 1)),
                                             fixed)
    B_path = jnp.concatenate([B_path, B[None]])
    M_path = jnp.concatenate([M_path, Mb[None]])
    risk = jax.vmap(lambda b, m: problem.risk(b, m, probs))(B_path, M_path)
    return np.arange(steps + 1) * dt, np.asarray(B_path), np.asarray(M_path), np.asarray(risk)


# --------------------------------------------------------------------------- #
# Streaming Adam on mixture data
# --------------------------------------------------------------------------- #
def sample_mixture(key, cov_sqrt, means, probs):
    k_c, k_x = jax.random.split(key)
    c = jax.random.categorical(k_c, jnp.log(probs))
    x = means[:, c] + cov_sqrt * jax.random.normal(k_x, cov_sqrt.shape, means.dtype)
    return x, c


def run_adam(problem: GMMProblem, theta0, cov_diag, means, probs, *, lr, beta1,
             beta2, eps, num_steps, record_every, keys):
    """Streaming batch-one Adam (no bias correction) vmapped over ``keys``.

    ``lr`` is the *rescaled* profile: step ``k`` uses ``lr(k / D) / D``.  The
    stability array is ``eps * K_ii`` as in the main paper.  Returns the order
    parameters recorded *before* steps ``0, record_every, 2 record_every, ...``
    plus the final iterate: ``B`` of shape ``(runs, R, M, M)`` and ``Mbar`` of
    shape ``(runs, R, M, C)``.
    """
    D, M = theta0.shape
    cov_sqrt = jnp.sqrt(cov_diag)
    eps_arr = eps * cov_diag[:, None]
    lr_at = lr if callable(lr) else (lambda t: lr)

    def order_params(theta):
        return theta.T @ (cov_diag[:, None] * theta), theta.T @ means

    def one_run(key):
        def step(carry, k):
            theta, m, v, key = carry
            key, sub = jax.random.split(key)
            x, c = sample_mixture(sub, cov_sqrt, means, probs)
            g = jnp.outer(x, problem.f(theta.T @ x, c))
            m = beta1 * m + (1.0 - beta1) * g
            v = beta2 * v + (1.0 - beta2) * g ** 2
            theta = theta - (lr_at(k / D) / D) * m / jnp.sqrt(v + eps_arr)
            return (theta, m, v, key), None

        def chunk(carry, start):
            obs = order_params(carry[0])
            carry, _ = lax.scan(step, carry, start + jnp.arange(record_every))
            return carry, obs

        zeros = jnp.zeros_like(theta0)
        num_records = num_steps // record_every
        carry, (Bs, Ms) = lax.scan(chunk, (theta0, zeros, zeros, key),
                                   jnp.arange(num_records) * record_every)
        Bf, Mf = order_params(carry[0])
        return jnp.concatenate([Bs, Bf[None]]), jnp.concatenate([Ms, Mf[None]])

    return jax.jit(jax.vmap(one_run))(keys)


def risks_from_paths(problem: GMMProblem, B, Mbar, probs):
    """Evaluate ``h`` and the classification error on recorded order parameters."""
    shape = B.shape[:-2]
    Bf = B.reshape((-1,) + B.shape[-2:])
    Mf = Mbar.reshape((-1,) + Mbar.shape[-2:])
    risk = jax.lax.map(lambda bm: problem.risk(bm[0], bm[1], probs), (Bf, Mf),
                       batch_size=256)
    err = jax.lax.map(lambda bm: problem.error(bm[0], bm[1], probs), (Bf, Mf),
                      batch_size=64)
    return np.asarray(risk).reshape(shape), np.asarray(err).reshape(shape)


# --------------------------------------------------------------------------- #
# Frozen-parameter check of the first moment (drift)
# --------------------------------------------------------------------------- #
def predicted_drift(theta, cov_diag, means, coeffs):
    """HAdam drift per unit rescaled time, divided by ``-eta``:
    ``Kbar Theta phi + Dinv Xbar chi`` (diagonal ``K``)."""
    phi, chi, _ = coeffs
    lam = jnp.sqrt(cov_diag)
    return lam[:, None] * (theta @ phi) + (means / lam[:, None]) @ chi


def frozen_adam_mean_update(problem: GMMProblem, theta, cov_diag, means, probs, *,
                            beta1, beta2, eps, num_steps, burn_in, key,
                            num_chains=8):
    """Average Adam direction ``m / sqrt(v + eps)`` with ``Theta`` held fixed.

    Runs ``num_chains`` independent chains for ``burn_in + num_steps`` steps and
    averages the direction over the last ``num_steps``; multiplying by ``-lr / D``
    gives the mean per-step update whose ``D``-step sum is the drift.
    """
    cov_sqrt = jnp.sqrt(cov_diag)
    eps_arr = eps * cov_diag[:, None]

    def chain(key):
        def step(carry, k):
            m, v, acc, key = carry
            key, sub = jax.random.split(key)
            x, c = sample_mixture(sub, cov_sqrt, means, probs)
            g = jnp.outer(x, problem.f(theta.T @ x, c))
            m = beta1 * m + (1.0 - beta1) * g
            v = beta2 * v + (1.0 - beta2) * g ** 2
            acc = acc + jnp.where(k >= burn_in, m / jnp.sqrt(v + eps_arr), 0.0)
            return (m, v, acc, key), None

        zeros = jnp.zeros_like(theta)
        (_, _, acc, _), _ = lax.scan(step, (zeros, zeros, zeros, key),
                                     jnp.arange(burn_in + num_steps))
        return acc / num_steps

    return jax.jit(jax.vmap(chain))(jax.random.split(key, num_chains))
