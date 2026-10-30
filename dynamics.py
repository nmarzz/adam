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

from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import lax

from discounts import (adam_noise_field, adam_ode_diffusion, compute_H, compute_I,
                       cov_from_B, drift_matrix_from_B, stationary_coefficients)
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
        # Kbar = D^{-1} K has right eigenvectors mu_j = D^{-1/2} w_j and left
        # eigenvectors rho_j = D^{1/2} w_j (w_j eigenvectors of D^{-1/2} K D^{-1/2}),
        # and p_j = Theta^T K Pi_j Theta with Pi_j = mu_j rho_j^T.
        scale = jnp.sqrt(jnp.diag(cov))
        root = jnp.sqrt(scale)
        eigs, w = jnp.linalg.eigh(cov / root[:, None] / root[None, :])
        A = cov @ (w / root[:, None])       # = K mu
        R = root[:, None] * w               # = rho
        p, u, q = _modes(A, R, params, optimal_params)
        corr = cov / scale[:, None] / scale[None, :]
        diff_data = ("dense", A, R, jnp.linalg.cholesky(corr))
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
                 history_length=500, diffusion_samples=2000,
                 diffusion_history=50, noise_samples=2000, noise_history=50,
                 noise_std=0.0,
                 return_B=False, key):
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
        key, k_phi, k_cov = jax.random.split(key, 3)
        V = drift_matrix_from_B(
            B, problem.f, beta1, beta2, k_phi,
            num_samples=num_samples, history_length=history_length,
            eps=eps, noise_std=noise_std,
        )
        p, u, q = y[:d], y[d:2 * d], y[2 * d:]
        lr_t = _lr_at(lr, i * dt)
        e = eigs[:, None, None]

        if diagonal:
            var_force = diff_data[1]
            sigma = cov_from_B(B, problem.f, beta1, beta2, k_cov,
                               num_samples=diffusion_samples,
                               history_length=diffusion_history, eps=eps,
                               noise_std=noise_std)
            diffusion = var_force[:, None, None] * sigma            # (D, M, M)
        else:
            _, KL, R, cov_chol = diff_data
            diffusion = adam_ode_diffusion(B, problem.f, beta1, beta2, k_cov, cov_chol,
                                           KL, R, num_samples=noise_samples,
                                           history_length=noise_history, eps=eps,
                                           noise_std=noise_std)

        pu = jnp.concatenate([p, u], axis=2)
        pv = jnp.einsum("dij,jk->dik", pu, V)
        p_up = -lr_t * e * (pv + jnp.swapaxes(pv, 1, 2))
        p_up = p_up + lr_t ** 2 * diffusion / d
        uq = jnp.concatenate([u, q], axis=1)
        u_up = -lr_t * e * jnp.einsum("ij,djk->dik", V.T, uq)
        y = y + dt * jnp.concatenate([p_up, u_up, jnp.zeros_like(u_up)])
        return (y, key), (risk, B)

    (_, _), (risks, B_path) = lax.scan(step, (y0, key), jnp.arange(iters))
    times = jnp.arange(iters) * dt
    if return_B:
        return risks, B_path, times
    return risks, times


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
                 noise_samples=2000, noise_history=50, noise_std=0.0, key):
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
    corr_chol = jnp.ones(1) if diagonal else jnp.linalg.cholesky(
        cov / jnp.sqrt(jnp.diag(cov))[:, None] / jnp.sqrt(jnp.diag(cov))[None, :]
    )
    d, m = params.shape
    iters = int(T / dt)

    def step(carry, i):
        params, key = carry
        B = make_B(params, optimal_params, cov)
        risk = problem.risk_from_B(B)
        key, k_phi, k_noise = jax.random.split(key, 3)
        V = drift_matrix_from_B(
            B, problem.f, beta1, beta2, k_phi,
            num_samples=num_samples, eps=eps, noise_std=noise_std,
        )
        stacked = jnp.concatenate([params, optimal_params], axis=1)
        if diagonal:
            mean = covbar[:, None] * (stacked @ V)
        else:
            mean = covbar @ stacked @ V

        if diagonal:
            k_cov, k_w = jax.random.split(k_noise)
            W = jax.random.normal(k_w, optimal_params.shape) * jnp.sqrt(dt)
            sigma = cov_from_B(B, problem.f, beta1, beta2, k_cov,
                               num_samples=num_samples, eps=eps,
                               noise_std=noise_std)
            noise = W @ jnp.linalg.cholesky(sigma)
        else:
            field = adam_noise_field(B, problem.f, beta1, beta2, k_noise, corr_chol,
                                     num_samples=noise_samples,
                                     history_length=noise_history, eps=eps,
                                     noise_std=noise_std)
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


# --------------------------------------------------------------------------- #
# Mode ODEs for mixture data, mini-batches and general (dense) K
# (gmm_adam.tex eq. gmm_system)
# --------------------------------------------------------------------------- #
class Modes(NamedTuple):
    """Spectral data of the deterministic equivalent, one row per mode ``j``.

    ``Kbar = sum_j nu_j mu_j rho_j^T`` with left/right eigenvectors
    ``rho_j = D^{1/2} w_j``, ``mu_j = D^{-1/2} w_j`` (``w_j`` orthonormal
    eigenvectors of ``D^{-1/2} K D^{-1/2}``), so ``K Pi_j = nu_j rho_j rho_j^T``,
    ``p_j = nu_j (rho_j^T Theta)(rho_j^T Theta)^T``, ``o_j = (Theta^T rho_j)(mu_j^T Xbar)``.

    The Ito term of mode ``j`` is ``(sigma - tau) w_iso_j + tau w_cov_j`` with
    ``w_iso_j = tr(K Pi_j)`` and ``w_cov_j = <C, K Pi_j>``, ``C`` the correlation
    matrix of the coordinates seen by the optimizer: this is
    ``<Sigma, K Pi_j>`` for ``Sigma_ik = sigma 1{i=k} + tau C_ik`` (``i != k``),
    exact for diagonal ``K`` and to first order in the off-diagonal
    correlations otherwise (e.g. Haar-rotated ``K``).
    """
    nu: jnp.ndarray        # (J,)
    student: jnp.ndarray   # (J, M)  rho_j^T Theta_0
    teacher: jnp.ndarray   # (J, M*) rho_j^T Theta*
    means: jnp.ndarray     # (J, C)  mu_j^T Xbar
    w_iso: jnp.ndarray     # (J,)
    w_cov: jnp.ndarray     # (J,)


def adam_modes(cov, params, teacher, means):
    """Modes of ``Kbar = diag(K)^{-1/2} K`` for a diagonal (``(D,)``) or dense ``K``.

    Also returns the left eigenvectors ``rho`` (``None`` for diagonal ``K``,
    where they are scaled coordinate vectors) so per-mode statistics of an
    optimizer run can be compared with the ODE.
    """
    if cov.ndim == 1:
        quarter = cov ** 0.25
        return Modes(jnp.sqrt(cov), quarter[:, None] * params,
                     quarter[:, None] * teacher, means / quarter[:, None],
                     cov, cov), None
    scale = jnp.sqrt(jnp.diag(cov))
    root = jnp.sqrt(scale)
    nu, w = jnp.linalg.eigh(cov / root[:, None] / root[None, :])
    rho, mu = root[:, None] * w, w / root[:, None]
    return Modes(nu, rho.T @ params, rho.T @ teacher, mu.T @ means,
                 nu * (scale @ w ** 2), nu ** 2), rho


def sgd_modes(cov, params, teacher, means):
    """Modes of ``K`` itself: SGD sees unnormalized coordinates (``D = Id``)."""
    if cov.ndim == 1:
        return Modes(cov, params, teacher, means, cov, cov ** 2), None
    lam, w = jnp.linalg.eigh(cov)
    return Modes(lam, w.T @ params, w.T @ teacher, w.T @ means, lam, lam ** 2), w


def _mode_state(modes):
    a, b, m, nu = modes.student, modes.teacher, modes.means, modes.nu[:, None, None]
    outer = lambda x, y: jnp.einsum("ja,jb->jab", x, y)
    state = (nu * outer(a, a), nu * outer(a, b), outer(a, m))
    constants = (nu * outer(b, b), outer(b, m), outer(m, m))
    return state, constants


def isotropic_modes(params, teacher, means):
    """Mode-summed state for ``K = Id`` (``Kbar = Id``: every ``nu_j = 1`` and
    ``w_iso_j = w_cov_j = 1``), where summing the mode ODEs over ``j`` is exact.

    Returns ``(modes, initial)`` for :func:`run_mixture_ode` (``initial=``) or
    :func:`run_mode_ode_logtime`: one aggregated mode with ``P = Theta^T Theta``,
    ``U = Theta^T Theta*``, ``O = Theta^T Xbar`` (no rank-one outer products),
    ``nu = 1`` and ``w_iso = w_cov = D``, so the Ito term is ``eta^2 sigma``.
    """
    d = params.shape[0]
    state = (params.T @ params, params.T @ teacher, params.T @ means)
    constants = (teacher.T @ teacher, teacher.T @ means, means.T @ means)
    lift = lambda x: x[None]
    modes = Modes(jnp.ones(1), jnp.zeros((1, params.shape[1])),
                  jnp.zeros((1, teacher.shape[1])), jnp.zeros((1, means.shape[1])),
                  jnp.full(1, float(d)), jnp.full(1, float(d)))
    return modes, (tuple(map(lift, state)), tuple(map(lift, constants)))


def _mode_field(state, constants, nu, w_iso, w_cov, co, eta, dimension):
    """Right-hand side of eq. gmm_system for every mode (``nu``, ``w`` are ``(J,1,1)``)."""
    P, U, Ob = state
    N, Os, G = constants
    phi, chi, sigma, tau = co["phi"], co["chi"], co["sigma"], co["tau"]
    drift = (jnp.einsum("jab,bc->jac", jnp.concatenate([P, U], axis=2), phi)
             + jnp.einsum("jac,cd->jad", Ob, chi))
    dP = (-eta * nu * (drift + jnp.swapaxes(drift, 1, 2))
          + eta ** 2 / dimension * ((sigma - tau) * w_iso + tau * w_cov))
    dU = -eta * nu * (jnp.einsum("ba,jbc->jac", phi, jnp.concatenate([U, N], axis=1))
                      + jnp.einsum("ca,jdc->jad", chi, Os))
    dOb = -eta * (nu * jnp.einsum("ba,jbc->jac", phi, jnp.concatenate([Ob, Os], axis=1))
                  + jnp.einsum("ca,jcd->jad", chi, G))
    return (dP, dU, dOb)


def _order_parameters(state, constants):
    P, U, Ob = state
    N, Os, _ = constants
    B = jnp.block([[P.sum(0), U.sum(0)], [U.sum(0).T, N.sum(0)]])
    return B, jnp.concatenate([Ob.sum(0), Os.sum(0)], axis=0)


@partial(jax.jit, static_argnames=["f", "schedule", "burn", "tail", "sgd",
                                   "steps", "record_every", "dimension", "stein"])
def _mixture_ode_scan(modes, lr, priors, draws, beta1, beta2, eps, dt, initial, *, f,
                      schedule, burn, tail, sgd, steps, record_every, dimension,
                      stein):
    state, constants = _mode_state(modes) if initial is None else initial
    N = constants[0]
    nu = modes.nu[:, None, None]
    w_iso, w_cov = modes.w_iso[:, None, None], modes.w_cov[:, None, None]

    def field(t, state):
        B, O = _order_parameters(state, constants)
        co = stationary_coefficients(B, O, priors, draws, f, beta1, beta2, eps,
                                     burn=burn, tail=tail, sgd=sgd, stein=stein)
        eta = lr * (schedule(t) if schedule is not None else 1.0)
        return _mode_field(state, constants, nu, w_iso, w_cov, co, eta, dimension)

    def heun(state, i):
        t = i * dt
        k1 = field(t, state)
        guess = jax.tree.map(lambda y, k: y + dt * k, state, k1)
        k2 = field(t + dt, guess)
        return jax.tree.map(lambda y, a, b: y + dt / 2 * (a + b), state, k1, k2), None

    def record(state):
        P, U, _ = state
        B, O = _order_parameters(state, constants)
        mode_error = jnp.trace(P - U - jnp.swapaxes(U, 1, 2) + N, axis1=1, axis2=2) \
            if P.shape[1] == N.shape[1] else jnp.zeros(P.shape[0])
        return B, O, mode_error

    def block(state, r):
        out = record(state)
        state, _ = lax.scan(heun, state, r * record_every + jnp.arange(record_every))
        return state, out

    state, (B, O, mode_error) = lax.scan(block, state, jnp.arange(steps // record_every))
    last = record(state)
    stack = lambda path, end: jnp.concatenate([path, end[None]])
    return stack(B, last[0]), stack(O, last[1]), stack(mode_error, last[2])


def run_mixture_ode(problem, modes, lr, T, *, beta1, beta2, eps, draws, layout,
                    dimension, dt=0.02, record_every=1, schedule=None, sgd=False,
                    stein=False, initial=None):
    """Integrate the mode ODEs (Heun) for a :class:`problems.MixtureProblem`.

    ``modes`` comes from :func:`adam_modes` (or :func:`sgd_modes` with
    ``sgd=True``); ``draws``/``layout`` from :func:`discounts.draw_paths` /
    :func:`discounts.path_layout` with the batch size of the optimizer. The
    learning rate is ``lr * schedule(t)``; ``stein`` selects the Stein-form
    estimator of ``phi`` (see :func:`discounts.stationary_coefficients`). Returns a dict with ``time``, ``B``,
    ``O``, ``risk``, ``error`` and ``mode_error`` (the per-mode excess
    ``tr(p_j - u_j - u_j^T + N_j)`` when ``M = M*``), recorded every
    ``record_every`` steps. ``initial`` (from :func:`isotropic_modes`, with its
    ``modes``) replaces the rank-one per-mode initial state.
    """
    steps = int(round(T / dt))
    steps -= steps % record_every
    B, O, mode_error = _mixture_ode_scan(
        modes, lr, problem.priors, draws, beta1, beta2, eps, dt, initial, f=problem.f,
        schedule=schedule, burn=layout[0], tail=layout[2], sgd=sgd, steps=steps,
        record_every=record_every, dimension=dimension, stein=stein)
    risk = jax.vmap(problem.risk)(B, O)
    error = jax.vmap(problem.error)(B, O) if problem.error is not None else None
    time = jnp.arange(len(B)) * dt * record_every
    return {"time": time, "B": B, "O": O, "risk": risk, "error": error,
            "mode_error": mode_error}


@partial(jax.jit, static_argnames=["coefficients", "schedule", "dimension", "safety",
                                   "max_steps"])
def _logtime_scan(aux, modes, initial, lr, times, dt_max, *, coefficients, schedule,
                  dimension, safety, max_steps):
    state, constants = initial
    nu = modes.nu[:, None, None]
    w_iso, w_cov = modes.w_iso[:, None, None], modes.w_cov[:, None, None]
    gram = jnp.linalg.norm(constants[2].sum(0), 2) if constants[2].size else 0.0
    eta_at = lambda t: lr * (schedule(t) if schedule is not None else 1.0)

    def evaluate(t, state):
        B, O = _order_parameters(state, constants)
        co = coefficients(B, O, aux)
        return co, _mode_field(state, constants, nu, w_iso, w_cov, co, eta_at(t), dimension)

    def interval(carry, t_next):
        def cond(c):
            t, _, n = c
            return (t < t_next) & (n < max_steps)

        def body(c):
            t, state, n = c
            co, k1 = evaluate(t, state)
            rate = eta_at(t) * jnp.max(jnp.sum(jnp.abs(co["phi"]), axis=0)) * (1.0 + gram)
            dt = jnp.minimum(jnp.minimum(t_next - t, safety / rate), dt_max)
            guess = jax.tree.map(lambda y, k: y + dt * k, state, k1)
            _, k2 = evaluate(t + dt, guess)
            state = jax.tree.map(lambda y, a, b: y + dt / 2 * (a + b), state, k1, k2)
            return t + dt, state, n + 1

        t, state, n = carry
        t, state, n_new = lax.while_loop(cond, body, (t, state, jnp.zeros((), int)))
        co, _ = evaluate(t, state)
        B, O = _order_parameters(state, constants)
        diag = jnp.stack([jnp.diag(co["phi"]), jnp.diag(co["chi"]), jnp.diag(co["sigma"])])
        return (t, state, n + n_new), (B, O, diag, n_new)

    t0 = jnp.asarray(times[0], dtype=state[0].dtype)
    B0, O0 = _order_parameters(state, constants)
    co0 = coefficients(B0, O0, aux)
    _, (B, O, diag, steps) = lax.scan(interval, (t0, state, jnp.zeros((), int)), times[1:])
    diag0 = jnp.stack([jnp.diag(co0["phi"]), jnp.diag(co0["chi"]), jnp.diag(co0["sigma"])])
    stack = lambda first, rest: jnp.concatenate([first[None], rest])
    return stack(B0, B), stack(O0, O), stack(diag0, diag), steps


def run_mode_ode_logtime(coefficients, aux, modes, initial, lr, times, *, dimension,
                         schedule=None, safety=0.2, dt_max=jnp.inf, max_steps=100_000):
    """Mode ODEs with adaptive Heun steps, recorded at ``times`` (e.g. log-spaced).

    ``coefficients(B, O, aux) -> {"phi", "chi", "sigma", "tau"}`` is a static
    callable and ``aux`` its array arguments (e.g. a :class:`discounts.SoftmaxPlan`).
    Each step uses ``dt = min(safety / (eta ||phi||_1 (1 + ||G||)), dt_max)``,
    shortened to land on every record time, so steps grow as the dynamics
    slow down. Returns ``B``, ``O``, the diagonals of ``(phi, chi, sigma)``
    (shape ``(len(times), 3, M)``) and the number of steps per interval.
    """
    times = jnp.asarray(times, dtype=float)
    B, O, diag, steps = _logtime_scan(aux, modes, initial, lr, times, dt_max,
                                      coefficients=coefficients, schedule=schedule,
                                      dimension=dimension, safety=safety,
                                      max_steps=max_steps)
    return {"time": times, "B": B, "O": O, "coef_diag": diag, "steps": steps}
