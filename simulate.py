"""Functional, ``lax.scan``-driven simulators for the streaming optimizers.

Each optimizer is built as a :class:`Sim` (an ``init`` / ``step`` pair of pure
closures) by a ``build_*`` factory that closes over the problem, the data
covariance, the teacher, the learning rate, and the hyper-parameters. A single
:func:`run` driver scans the step over ``num_steps``; :func:`run_many` ``vmap``\\s
that scan over a batch of PRNG keys for confidence bands.

Replaces the class-based ``optimizers.py``: no Python time-loops, no jitted
methods mutating ``self``, and Block Adam (``main.tex`` eq. block_adam) is
implemented for the first time.
"""
from __future__ import annotations

import math
from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp
from jax import lax

from problems import Problem
from utils import make_B, make_data


class Sim(NamedTuple):
    init: Callable   # params -> state pytree
    step: Callable   # (params, state, key, k) -> (params, state)


def _lr_at(lr, k):
    return lr(k) if callable(lr) else lr


def _sample_grad(problem: Problem, cov, optimal_params, params, key,
                 label_noise_std=0.0):
    """One fresh streaming gradient at ``params``."""
    data = make_data(cov, key)
    target = problem.target(optimal_params, data)
    if label_noise_std:
        key_noise = jax.random.fold_in(key, 1)
        target = target + label_noise_std * jax.random.normal(key_noise, target.shape)
    return problem.grad(params, data, target)


# --------------------------------------------------------------------------- #
# Optimizer factories
# --------------------------------------------------------------------------- #
def build_adam(problem, cov, optimal_params, lr, *, beta1, beta2, eps=0.0,
               label_noise_std=0.0) -> Sim:
    def init(params):
        return (jnp.zeros_like(params), jnp.zeros_like(params))

    def step(params, state, key, k):
        m, v = state
        g = _sample_grad(
            problem, cov, optimal_params, params, key, label_noise_std
        )
        m = beta1 * m + (1 - beta1) * g
        v = beta2 * v + (1 - beta2) * g ** 2
        params = params - _lr_at(lr, k) * m / jnp.sqrt(v + eps)
        return params, (m, v)

    return Sim(init, step)


def build_block_adam(problem, cov, optimal_params, lr, *, beta1, beta2, eps=0.0, N=None) -> Sim:
    """Block Adam (``main.tex`` eq. block_adam).

    The moment estimates reset at every block boundary ``k % N == 0`` and, within
    a block, all gradients are evaluated at the frozen reference parameter
    ``theta_{floor(k/N) N}``. ``N`` grows with the ambient dimension; default
    ``round(sqrt(D))`` (matching the old ``MBlockSGD``).
    """
    d = optimal_params.shape[0]
    block = int(N) if N is not None else max(1, round(math.sqrt(d)))

    def init(params):
        return (jnp.zeros_like(params), jnp.zeros_like(params), params)  # m, v, ref

    def step(params, state, key, k):
        m, v, ref = state
        at_start = (k % block) == 0
        ref = jnp.where(at_start, params, ref)            # snapshot theta at block start
        keep = jnp.where(at_start, 0.0, 1.0)              # indicator k != nround(k)
        g = _sample_grad(problem, cov, optimal_params, ref, key)
        m = beta1 * m * keep + (1 - beta1) * g
        v = beta2 * v * keep + (1 - beta2) * g ** 2
        params = params - _lr_at(lr, k) * m / jnp.sqrt(v + eps)
        return params, (m, v, ref)

    return Sim(init, step)


def build_sgd(problem, cov, optimal_params, lr) -> Sim:
    def init(params):
        return ()

    def step(params, state, key, k):
        g = _sample_grad(problem, cov, optimal_params, params, key)
        params = params - _lr_at(lr, k) * g
        return params, ()

    return Sim(init, step)


def build_block_sgd(problem, cov, optimal_params, lr, *, N=None) -> Sim:
    """Block SGD: gradients use a reference parameter frozen for ``N`` steps.

    The ``MBlockSGD`` rewrite — the reference now lives in the scan carry instead
    of a ``self`` attribute mutated inside a jitted method.
    """
    d = optimal_params.shape[0]
    block = int(N) if N is not None else max(1, round(math.sqrt(d)))

    def init(params):
        return (params,)  # ref

    def step(params, state, key, k):
        (ref,) = state
        g = _sample_grad(problem, cov, optimal_params, ref, key)
        params = params - _lr_at(lr, k) * g
        refresh = ((k + 1) % block) == 0
        ref = jnp.where(refresh, params, ref)
        return params, (ref,)

    return Sim(init, step)


def build_sgd_momentum(problem, cov, optimal_params, lr, *, beta1) -> Sim:
    def init(params):
        return (jnp.zeros_like(params),)  # m

    def step(params, state, key, k):
        (m,) = state
        g = _sample_grad(problem, cov, optimal_params, params, key)
        m = beta1 * m + (1 - beta1) * g
        params = params - _lr_at(lr, k) * m
        return params, (m,)

    return Sim(init, step)


def build_block_sgd_momentum(problem, cov, optimal_params, lr, *, beta1, N=None) -> Sim:
    """Block SGD with momentum: ``build_block_adam`` without the second moment ``v``.

    Same block structure as :func:`build_block_adam` (eq. block_adam): ``m`` resets
    and the gradient reference snapshots at every block boundary ``k % N == 0``,
    and within a block all gradients are evaluated at the frozen reference
    ``theta_{floor(k/N) N}``.
    """
    d = optimal_params.shape[0]
    block = int(N) if N is not None else max(1, round(math.sqrt(d)))

    def init(params):
        return (jnp.zeros_like(params), params)  # m, ref

    def step(params, state, key, k):
        m, ref = state
        at_start = (k % block) == 0
        ref = jnp.where(at_start, params, ref)
        keep = jnp.where(at_start, 0.0, 1.0)
        g = _sample_grad(problem, cov, optimal_params, ref, key)
        m = beta1 * m * keep + (1 - beta1) * g
        params = params - _lr_at(lr, k) * m
        return params, (m, ref)

    return Sim(init, step)


def build_resampled_adam(problem, cov, optimal_params, lr, *, beta1, beta2,
                         eps=0.0, history_length=15) -> Sim:
    """Resampled Adam: at each step the moment estimates are built from
    ``history_length`` freshly-resampled gradients (vectorized; the old nested
    Python loops did ``history_length**2`` gradient evaluations per step)."""
    H = history_length
    decay1 = beta1 ** jnp.arange(H) * (1 - beta1)
    decay2 = beta2 ** jnp.arange(H) * (1 - beta2)

    def init(params):
        return ()

    def step(params, state, key, k):
        key_cur, key_hist = jax.random.split(key)
        current = _sample_grad(problem, cov, optimal_params, params, key_cur)
        current2 = current ** 2

        hist_keys = jax.random.split(key_hist, H)
        grads = jax.vmap(
            lambda kk: _sample_grad(problem, cov, optimal_params, params, kk)
        )(hist_keys)                                   # (H, d, m)
        base2 = grads ** 2

        S = jnp.einsum("i,idm->dm", decay2, base2)     # full decayed second moment
        # per-slot accumulator: slot l replaced by the current gradient
        sm = jnp.sqrt(S[None] + decay2[:, None, None] * (current2[None] - base2))
        contributions = current[None] / sm             # (H, d, m)
        update = jnp.einsum("i,idm->dm", decay1, contributions)
        params = params - _lr_at(lr, k) * update
        return params, ()

    return Sim(init, step)


# --------------------------------------------------------------------------- #
# Drivers
# --------------------------------------------------------------------------- #
def run(sim: Sim, problem: Problem, params0, optimal_params, cov, num_steps, *, key):
    """Scan ``sim`` for ``num_steps`` streaming steps. Returns (final_params, risks).

    ``risks[k]`` is the risk *before* the ``k``-th update, so ``risks[0]`` is the
    risk at initialization (matches the old ``Optimizer.run`` convention)."""
    def scan_step(carry, k):
        params, state, key = carry
        risk = problem.risk_from_B(make_B(params, optimal_params, cov))
        key, sub = jax.random.split(key)
        params, state = sim.step(params, state, sub, k)
        return (params, state, key), risk

    init_carry = (params0, sim.init(params0), key)
    (params, _, _), risks = lax.scan(scan_step, init_carry, jnp.arange(num_steps))
    return params, risks


def run_many(sim: Sim, problem: Problem, params0, optimal_params, cov, num_steps, *, keys):
    """``vmap`` :func:`run` over a batch of PRNG keys -> risks of shape
    ``(n_seeds, num_steps)``."""
    def one(key):
        _, risks = run(sim, problem, params0, optimal_params, cov, num_steps, key=key)
        return risks

    return jax.vmap(one)(keys)


def run_with_B(sim: Sim, problem: Problem, params0, optimal_params, cov,
               num_steps, *, key):
    """Run a simulator and return ``(final_params, risks, B_path)``.

    Like :func:`run`, observables are recorded immediately before each update.
    ``B_path`` has shape ``(num_steps, M+M*, M+M*)``.
    """
    def scan_step(carry, k):
        params, state, key = carry
        B = make_B(params, optimal_params, cov)
        risk = problem.risk_from_B(B)
        key, sub = jax.random.split(key)
        params, state = sim.step(params, state, sub, k)
        return (params, state, key), (risk, B)

    init_carry = (params0, sim.init(params0), key)
    (params, _, _), (risks, B_path) = lax.scan(
        scan_step, init_carry, jnp.arange(num_steps)
    )
    return params, risks, B_path


def run_many_with_B(sim: Sim, problem: Problem, params0, optimal_params, cov,
                    num_steps, *, keys):
    """Vectorize :func:`run_with_B` over keys.

    Returns ``(risks, B_paths)`` with leading dimension ``len(keys)``.
    """
    def one(key):
        _, risks, B_path = run_with_B(
            sim, problem, params0, optimal_params, cov, num_steps, key=key
        )
        return risks, B_path

    return jax.vmap(one)(keys)


# --------------------------------------------------------------------------- #
# Mini-batch optimizers on Gaussian-mixture data (gmm_adam.tex)
# --------------------------------------------------------------------------- #
class MixtureData(NamedTuple):
    """``x | c ~ N(means[:, c], K)`` with ``K = factor**2`` (diagonal, ``(D,)``)
    or ``K = factor @ factor.T`` (dense, ``(D, D)``; e.g. ``O diag(sqrt(lam))``
    for a Haar-rotated spectrum). Centered data use one zero-mean class."""
    factor: jnp.ndarray    # (D,) or (D, D)
    means: jnp.ndarray     # (D, C)
    teacher: jnp.ndarray   # (D, M*)


def mixture_covariance(data: MixtureData):
    """``K`` as a ``(D,)`` diagonal or a dense ``(D, D)`` matrix."""
    f = data.factor
    return f ** 2 if f.ndim == 1 else f @ f.T


def _mixture_grad(problem, data, params, key, batch):
    """Batch-averaged gradient ``b^{-1} sum_beta x_beta (x) f(q_beta; c_beta, w_beta)``."""
    k_c, k_z, k_w = jax.random.split(key, 3)
    d = data.means.shape[0]
    classes = jax.random.categorical(k_c, jnp.log(problem.priors), shape=(batch,))
    z = jax.random.normal(k_z, (batch, d))
    noise = z * data.factor if data.factor.ndim == 1 else z @ data.factor.T
    x = data.means.T[classes] + noise
    w = jax.random.normal(k_w, (batch, problem.outputs))
    q = x @ jnp.concatenate([params, data.teacher], axis=1)
    return x.T @ jax.vmap(problem.f)(q, classes, w) / batch


def build_mixture_adam(problem, data: MixtureData, lr, *, beta1, beta2, eps=0.0,
                       batch=1) -> Sim:
    """Adam with batch size ``batch`` and stability array ``eps * K_ii``."""
    cov = mixture_covariance(data)
    stability = eps * (cov if cov.ndim == 1 else jnp.diag(cov))[:, None]

    def init(params):
        return (jnp.zeros_like(params), jnp.zeros_like(params))

    def step(params, state, key, k):
        m, v = state
        g = _mixture_grad(problem, data, params, key, batch)
        m = beta1 * m + (1 - beta1) * g
        v = beta2 * v + (1 - beta2) * g ** 2
        params = params - _lr_at(lr, k) * m / jnp.sqrt(v + stability)
        return params, (m, v)

    return Sim(init, step)


def build_mixture_sgd(problem, data: MixtureData, lr, *, batch=1) -> Sim:
    def init(params):
        return ()

    def step(params, state, key, k):
        g = _mixture_grad(problem, data, params, key, batch)
        return params - _lr_at(lr, k) * g, ()

    return Sim(init, step)


def run_mixture(sim: Sim, data: MixtureData, params0, num_steps, *, record_every,
                key, basis=None):
    """Scan ``sim`` and record, every ``record_every`` steps (including step 0
    and the last step), ``B = (Theta^+)^T K Theta^+``, ``O = (Theta^+)^T Xbar``
    and, if ``basis`` (``(D, J)``) is given, ``basis^T Theta``."""
    cov = mixture_covariance(data)

    def observe(params):
        W = jnp.concatenate([params, data.teacher], axis=1)
        KW = cov[:, None] * W if cov.ndim == 1 else cov @ W
        modes = basis.T @ params if basis is not None else jnp.zeros((0,) + params.shape[1:])
        return W.T @ KW, W.T @ data.means, modes

    def inner(carry, k):
        params, state, key = carry
        key, sub = jax.random.split(key)
        params, state = sim.step(params, state, sub, k)
        return (params, state, key), None

    def outer(carry, r):
        out = observe(carry[0])
        carry, _ = lax.scan(inner, carry, r * record_every + jnp.arange(record_every))
        return carry, out

    carry = (params0, sim.init(params0), key)
    carry, (B, O, modes) = lax.scan(outer, carry, jnp.arange(num_steps // record_every))
    last = observe(carry[0])
    stack = lambda path, end: jnp.concatenate([path, end[None]])
    return stack(B, last[0]), stack(O, last[1]), stack(modes, last[2])


def run_mixture_many(sim: Sim, data: MixtureData, params0, num_steps, *,
                     record_every, keys, basis=None):
    """``vmap`` :func:`run_mixture` over PRNG keys (leading axis = run)."""
    return jax.vmap(lambda key: run_mixture(sim, data, params0, num_steps,
                                            record_every=record_every, key=key,
                                            basis=basis))(keys)
