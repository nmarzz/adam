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


def _sample_grad(problem: Problem, cov, optimal_params, params, key):
    """One fresh streaming gradient at ``params``."""
    data = make_data(cov, key)
    target = problem.target(optimal_params, data)
    return problem.grad(params, data, target)


# --------------------------------------------------------------------------- #
# Optimizer factories
# --------------------------------------------------------------------------- #
def build_adam(problem, cov, optimal_params, lr, *, beta1, beta2, eps=0.0) -> Sim:
    def init(params):
        return (jnp.zeros_like(params), jnp.zeros_like(params))

    def step(params, state, key, k):
        m, v = state
        g = _sample_grad(problem, cov, optimal_params, params, key)
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
