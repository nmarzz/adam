"""Frozen psi-route coefficients for series J (Stage 1), without the series I driver's
side effects.

``psi_plan`` mirrors ``experiments/imbalance.py: adam_plan`` argument for argument (the
plan the cached series I ODEs used: ``C <= 32`` via ``zipf_scaling.make_plan`` defaults,
``C > 32`` single chunk ``2^27``), but does not call ``imbalance.jax_modules`` (which points
the persistent compilation cache at ``figs/experiments/imbalance/jaxcache``); this module
uses ``figs/experiments/knee/analysis/jaxcache`` instead. ``recheck_cached.py`` checks
the identity by reproducing the stored coefficient diagonals of a cached ODE state.
"""
from __future__ import annotations

import functools
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import diagnostics as D  # noqa: E402  (x64, no preallocation)

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

REPO = HERE.parents[2]
CACHE = REPO / "figs" / "experiments" / "knee" / "analysis" / "jaxcache"
CACHE.mkdir(parents=True, exist_ok=True)
jax.config.update("jax_compilation_cache_dir", str(CACHE))
jax.config.update("jax_persistent_cache_min_compile_time_secs", 5.0)
import discounts  # noqa: E402

BETA1, EPS, BATCH = 0.9, 1e-8, 1


@functools.lru_cache(maxsize=None)
def _plan(priors_key, beta2, seed):
    priors = jnp.asarray(np.array(priors_key))
    key = jax.random.PRNGKey(seed)
    window = 1024 if beta2 <= 0.99 else 8192
    if len(priors) <= 32:
        return discounts.softmax_psi_plan(key, priors, BETA1, beta2, batch=BATCH,
                                          plain_batches=max(2 ** 14, 2 ** 18 // BATCH),
                                          window=window)
    return discounts.softmax_psi_plan(key, priors, BETA1, beta2, batch=BATCH,
                                      plain_batches=max(2 ** 14, 2 ** 18 // BATCH),
                                      window=window, chunk_elements=2 ** 27)


def psi_plan(priors, beta2, seed=0):
    return _plan(tuple(float(p) for p in np.asarray(priors)), float(beta2), int(seed))


def coefficients(B, O, priors, beta2, *, seed=0, eps=EPS, verbose=True):
    """Full ``phi``, ``chi``, ``sigma`` (numpy) at ``(B, O)`` with the series I plan."""
    layout, plan = psi_plan(priors, beta2, seed)
    t0 = time.time()
    co = discounts.softmax_psi_coefficients(jnp.asarray(B, dtype=jnp.float64),
                                            jnp.asarray(O, dtype=jnp.float64),
                                            jnp.asarray(priors), plan, BETA1, beta2, eps,
                                            layout=layout)
    out = {k: np.asarray(co[k]) for k in ("phi", "chi", "sigma")}
    if verbose:
        print(f"  psi coefficients C={len(priors)} beta2={beta2:g}: {time.time() - t0:.1f}s",
              flush=True)
    return out
