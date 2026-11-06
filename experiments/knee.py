#!/usr/bin/env python3
"""Series J: plateau -> cliff -> knee -> tail of Adam's per-class losses (direct simulations).

Plan: ``~/.claude/plans/looking-at-the-experiments-rustling-shore.md`` (Stage 2 engineering,
Stage 3 runs). Model as in series I (``experiments/imbalance/README.md``): ``C`` classes,
priors ``pi_c ~ c^{-alpha}`` (``alpha = 0``: uniform), means ``r q_c`` (``r = 5``), noise
``K = kappa Id``, softmax cross-entropy, no bias, batch 1, deterministic ``Theta_0`` of
``zs.zipf_model``. Adam: ``beta1 = 0.9``, K-scaled stability ``eps K_ii``, no bias
correction, ``eta = 0.3`` (step ``eta / D``); SGD ``eta = 3``.

Lane groups. One group per ``(opt, C, alpha, kappa, eps, D, runs, seed, dtype, U)``: Adam
groups ``vmap`` the lanes ``beta2 x runs`` (lane ``b * runs + r``) through one executable,
SGD groups the runs. Run ``r`` uses ``key_r = fold_in(PRNGKey(seed), r)`` in every group
(common random numbers across ``beta2``, ``kappa``, ``alpha``, ``eps``; prefix-stable in
``runs``).

Records. Global nested grid ``steps_k = round(D 0.01 10^(k/16))``, integer ``k >= -6``,
with step 0 prepended (record index ``j``: ``j = 0`` is step 0, ``j >= 1`` is
``steps_{k = j - 7}``), independent of the horizon; chunk ``n`` ends at ``t = 0.01
10^(n/2)``. Every call of the chunk executable advances exactly
``H = 8`` records (half a decade), carry fed back, so a run extended from a checkpoint is
bitwise identical to a direct run. Chunk ``n`` holds records ``8n .. 8n + 7``
(``runs/<group>/chunk_{n:03d}.npz``); :func:`load_group` concatenates them.

Per chunk (``H`` records): per run and class ``loss, err`` (MC, float64, ``2^14`` fixed
draws: the estimator and draws of ``problems.softmax_class_metrics``, evaluated by the
faster :func:`class_metrics`), ``loss_cv`` (same draws with a control variate; use it for
small losses), ``Bdiag, Odiag``, ``margin_mean = O_cc - mean_{k!=c}
O_kc``, ``margin_min = O_cc - max_{k!=c} O_kc``, Adam column statistics ``vbar_c = mean_i
v_ic``, ``nu_c = mean_i |m_ic| / sqrt(v_ic + eps K_ii)``, ``frac_eps_c = mean_i [v_ic < eps
K_ii]`` and ``Udiag`` of ``U = Xbar^T (m / sqrt(v + eps K))`` (rows: class mean ``c``,
columns: logit ``k``; the parameter update moves ``O`` by ``-(eta/D) U^T``); NaN for SGD.
Seed means per ``beta2``: full ``B, O`` (float64) and ``U`` (float32). Array shapes
``(H, n_beta2, runs, C)`` and ``(H, n_beta2, C, C)``.

Subcommands: ``selftest | probe | run | status``. Outputs under ``figs/experiments/knee``:
``runs/<group>/{chunk_*.npz, ckpt/, meta.json}``, ``logs/<group>.log``, ``runs.jsonl``,
``status.jsonl``, ``probe.json``, ``jaxcache/``.
"""
from __future__ import annotations

import argparse
import functools
import hashlib
import inspect
import json
import math
import os
import runpy
import shutil
import sys
import time
from pathlib import Path
from typing import NamedTuple

import numpy as np

# a shared GPU: allocate on demand instead of grabbing 75% of the device
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
import imbalance as I  # noqa: E402  (numpy-only at import)

OUT = ROOT / "figs" / "experiments" / "knee"
R = I.R                 # 5.0
BETA1 = I.BETA1         # 0.9
ETA = I.ETA             # {"adam": 0.3, "sgd": 3.0}
BATCH = 1
SEED_J = 2026
H = 8                   # records per chunk (= half a decade at 16 per decade)
PER_DECADE = 16
T_MIN = 0.01
K_OFFSET = 7            # record j >= 1 is grid index k = j - 7 (chunk ends on half decades)
SAMPLES = 2 ** 16       # MC draws for the per-class metrics (2^14: 2-6% per-class error late)
CV_MAX_VAR = 6.0        # loss_cv uses the control variate only below this weighted s^2
BETA2S = (0.9, 0.95, 0.99, 0.999)
KAPPAS = (2.0, 1.0, 0.5, 0.25, 0.1)
STEP_CAP = 2e8          # cap on total steps (T_cap <= STEP_CAP / D)
A_S0 = 0.09             # SGD t_c(pi) ~ A_S(kappa) / pi, A_S = 0.09 kappa^0.4 (cached C=128 ODEs)


# --------------------------------------------------------------------------- #
# Groups, names, grid (numpy only)
# --------------------------------------------------------------------------- #
class Group(NamedTuple):
    opt: str
    classes: int
    alpha: float
    kappa: float
    eps: float
    dimension: int
    runs: int
    seed: int = SEED_J
    dtype: str = "f64"
    unroll: int = 1
    eta: float | None = None
    beta2s: tuple = BETA2S

    @property
    def lr(self):
        """``eta`` in ODE units (the step is ``lr / D``)."""
        return ETA[self.opt] if self.eta is None else float(self.eta)

    @property
    def lane_beta2(self):
        """``beta2`` lanes of the group (one placeholder lane for SGD)."""
        return tuple(self.beta2s) if self.opt == "adam" else (float("nan"),)

    @property
    def lanes(self):
        return len(self.lane_beta2) * self.runs


def default_dimension(kappa):
    """``D = max(1000, 20 r'^2)`` with ``r'^2 = 25 / kappa`` (finite-D lag ``r'^2/D <= 5%``)."""
    return int(max(1000, math.ceil(20 * R ** 2 / kappa - 1e-9)))


def name(g: Group):
    """``J_{opt}_C{C}_a{alpha}_k{kappa}_e{eps}_D{D}_R{runs}_s{seed}_{f64|f32}_u{U}[_b..][_eta]``.
    SGD has no ``eps`` (it does not use it), so its name omits ``_e`` and :func:`pinned`
    stores ``eps = None``; ``_b{beta2s}`` only for a non-default Adam lane set."""
    e = f"_e{g.eps:g}" if g.opt == "adam" else ""
    s = (f"J_{g.opt}_C{g.classes}_a{g.alpha:g}_k{g.kappa:g}{e}_D{g.dimension}_R{g.runs}"
         f"_s{g.seed}_{g.dtype}_u{g.unroll}")
    if g.opt == "adam" and tuple(float(b) for b in g.beta2s) != tuple(BETA2S):
        s += "_b" + "-".join(f"{float(b):g}" for b in g.beta2s)
    if g.eta is not None and float(g.eta) != ETA[g.opt]:
        s += f"_eta{float(g.eta):g}"
    return s


def record_steps(dimension, n):
    """First ``n`` record steps: ``[0] + [round(D t_k) for k = -6, -5, ...]`` with
    ``t_k = 0.01 10^(k/16)`` (record ``j >= 1`` is ``k = j - K_OFFSET``; nested; repeated
    steps give zero-length segments). ``k`` starts at ``-6`` so that chunk ``n`` ends
    exactly at ``k = 8 n``, ``t = 0.01 10^(n/2)`` (decades and half decades)."""
    k = np.arange(max(n - 1, 0)) + 1 - K_OFFSET
    steps = np.floor(dimension * T_MIN * 10.0 ** (k / PER_DECADE) + 0.5).astype(np.int64)
    return np.concatenate([[0], steps]).astype(np.int64)[:n]


def record_times(dimension, n):
    return record_steps(dimension, n) / dimension


def grid_index(T):
    """Smallest grid index ``k`` with ``0.01 10^(k/16) >= T``."""
    return max(1 - K_OFFSET, int(math.ceil(PER_DECADE * math.log10(T / T_MIN) - 1e-9)))


def snap(T):
    return T_MIN * 10.0 ** (grid_index(T) / PER_DECADE)


def chunks_for(T):
    """Chunks needed to record up to time ``T`` (record ``j = k + K_OFFSET``); chunk ``n``
    ends at ``t = 0.01 10^(n/2)``."""
    return int(math.ceil((grid_index(T) + K_OFFSET + 1) / H))


def chunk_records(n):
    return np.arange(n * H, (n + 1) * H)


def chunk_bounds(dimension, n):
    """``(H, 2)`` global ``[lo, hi)`` step bounds of chunk ``n`` (record ``j`` ends at
    ``steps[j]``; record 0 is the zero-length segment ``[0, 0)``)."""
    steps = record_steps(dimension, (n + 1) * H)
    prev = np.concatenate([[0], steps[:-1]])
    j = chunk_records(n)
    return np.stack([prev[j], steps[j]], 1).astype(np.int64)


def chunk_time(dimension, n):
    """Time of the last record of chunk ``n``."""
    return float(record_times(dimension, (n + 1) * H)[-1])


def horizon_rule(opt, classes, alpha, kappa, beta2s=BETA2S):
    """Initial horizon ``T0`` (time units): SGD ``10 t_S(pi_min)``, ``t_S = A_S/pi``,
    ``A_S = 0.09 kappa^0.4``; Adam that times ``max_beta2 max(window(pi_min), rare)`` from
    ``lib.ratio_asymptotes``. Snapped up onto the grid, at least 100. Also returns the
    cap ``min(100 t_A, STEP_CAP / D)`` with ``t_A = T0_raw / 10``."""
    pi_min = float(_priors(classes, alpha).min())
    t_s = A_S0 * kappa ** 0.4 / pi_min
    T0 = 10 * t_s
    if opt == "adam":
        sys.path.insert(0, str(HERE / "imbalance" / "analysis"))
        import lib  # noqa: E402  (numpy/scipy)
        ratio = 0.0
        for b2 in beta2s:
            window, rare = lib.ratio_asymptotes(b2, classes, kappa)
            ratio = max(ratio, float(window(pi_min)), float(rare))
        T0 *= ratio
    return max(snap(T0), 100.0), T0 / 10


def default_cap(t_char, dimension, T0=0.0):
    """``min(100 t_A, 2e8 / D)``, but never below the initial horizon ``T0``."""
    return max(min(100 * t_char, STEP_CAP / dimension), T0)


def _priors(classes, alpha):
    pi = np.arange(1, classes + 1, dtype=float) ** -float(alpha)
    return pi / pi.sum()


def git_hash():
    return I.git_hash()


NUMERICS_FILES = ("simulate.py", "problems.py", "utils.py", "experiments/zipf_scaling.py")


def device_info():
    """``{device, jaxlib}`` of the default backend (bitwise resume is per device)."""
    import jax
    import jaxlib
    d = jax.devices()[0]
    return {"device": f"{d.platform}:{getattr(d, 'device_kind', '?')}",
            "jaxlib": jaxlib.__version__}


def git_dirty():
    import subprocess
    try:
        out = subprocess.check_output(["git", "status", "--porcelain", "--", *NUMERICS_FILES,
                                       "experiments/knee.py", "experiments/imbalance.py"],
                                      cwd=ROOT, text=True)
        return bool(out.strip())
    except Exception:
        return None


# --------------------------------------------------------------------------- #
# Certificate (numpy only)
# --------------------------------------------------------------------------- #
def _ols_slope(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    xm = x - x.mean()
    return float(np.sum(xm * (y - y.mean())) / np.sum(xm ** 2))


def _first_crossing(t, y, level):
    """First time ``y <= level`` (log-linear interpolation in ``t``; NaN if never)."""
    idx = np.flatnonzero(y <= level)
    if len(idx) == 0:
        return np.nan
    i = int(idx[0])
    if i == 0 or t[i - 1] <= 0:
        return float(t[i])
    y0, y1 = np.log(y[i - 1]), np.log(y[i])
    f = (np.log(level) - y0) / (y1 - y0) if y1 != y0 else 1.0
    return float(np.exp(np.log(t[i - 1]) + f * (np.log(t[i]) - np.log(t[i - 1]))))


def _index_slope(t, logL, i0, i1):
    """OLS slope of ``log L`` on ``log t`` over records ``i0 .. i1`` inclusive (last axis =
    time). Windows are chosen by record index (16 per decade), not by comparing rounded
    record times, so a half decade is always 9 records and a decade 17."""
    if i0 < 0 or i1 - i0 < 1:
        return np.full(logL.shape[:-1], np.nan)
    tt = np.asarray(t[i0:i1 + 1], float)
    if (tt <= 0).any():
        return np.full(logL.shape[:-1], np.nan)
    x = np.log(tt) - np.log(tt).mean()
    y = logL[..., i0:i1 + 1]
    return np.sum(x * (y - y.mean(-1, keepdims=True)), -1) / np.sum(x ** 2)


def _window_slope(t, logL, lo, hi):
    """OLS slope of ``log L`` on ``log t`` over ``t in [lo, hi]`` (last axis = time)."""
    m = (t >= lo * (1 - 1e-12)) & (t <= hi * (1 + 1e-12)) & (t > 0)
    if m.sum() < 2:
        return np.full(logL.shape[:-1], np.nan)
    x = np.log(t[m]) - np.log(t[m]).mean()
    y = logL[..., m]
    return np.sum(x * (y - y.mean(-1, keepdims=True)), -1) / np.sum(x ** 2)


def _gap_stationarity(O0, O1):
    """Per class ``c``: ``max_k |g_kc(T) - g_kc(T0)| / max_k |g_kc(T)|`` with gaps
    ``g_kc = O_cc - O_kc`` (column ``c`` of ``lib.gap_stationarity``'s matrix)."""
    G = lambda o: np.diagonal(o, axis1=-2, axis2=-1)[..., None, :] - o
    a, b = G(O0), G(O1)
    return np.abs(b - a).max(-2) / np.maximum(np.abs(b).max(-2), 1e-300)


def certify(rec, *, at_cap=False, n_boot=200, seed=0, fixed_floor=1e-3, fixed_slope=0.02,
            tail_dslope=0.05, tail_se=0.05):
    """Per ``(beta2, class)`` convergence certificate from :func:`load_group` records.

    Seed means over runs, bootstrap SEs over runs (``n_boot`` resamples).

    * FIXED: gap stationarity (largest change of the seed-mean gaps ``O_cc - O_kc`` over
      ``[T/sqrt(10), T]`` / largest gap) below ``max(1e-3, 2 SE_runs)`` and
      ``|d log ell / d log t| < 0.02`` over ``[T/sqrt(10), T]``, both at ``T`` and at the
      previous check ``T/sqrt(10)`` (8 records earlier; checks are every half decade).
      ``SE_runs``: bootstrap SE of the same relative change of the per-run
      ``margin_mean`` (full per-run ``O`` is not stored).
    * TAIL: ``t_c(1/2) <= T/10``, the last-decade and previous-decade slopes of
      ``log ell`` on ``log t`` differ by less than 0.05, and the bootstrap SE of the
      last-decade slope is below 0.05.
    * MOVING otherwise; UNRESOLVED instead of MOVING when ``at_cap``.

    Returns a dict of ``(n_beta2, C)`` arrays plus ``status`` (str) and summary counts."""
    t = np.asarray(rec["time"], float)
    loss = np.asarray(rec["loss"], float)              # (N, nb, runs, C)
    mm = np.asarray(rec["margin_mean"], float)
    Om = np.asarray(rec["O_mean"], float)              # (N, nb, C, C)
    N, nb, runs, C = loss.shape
    rng = np.random.default_rng(seed)
    boot = rng.integers(0, runs, size=(n_boot, runs))

    Lm = loss.mean(2)                                  # (N, nb, C)
    logL = np.log(np.moveaxis(Lm, 0, -1))              # (nb, C, N)
    logLb = np.log(np.moveaxis(loss[:, :, boot].mean(3), 0, -1))   # (nb, n_boot, C, N)

    def fixed_at(n):
        k0 = n - H                                       # T / sqrt(10), half a decade back
        if k0 < 1:
            z = np.zeros((nb, C))
            return z.astype(bool), z + np.nan, z + np.nan, z + np.nan, z + np.nan
        slope = _index_slope(t, logL, k0, n)
        stat = _gap_stationarity(Om[k0], Om[n])                     # (nb, C)
        rel = lambda m0, m1: np.abs(m1 - m0) / np.maximum(np.abs(m1), 1e-300)
        sb = rel(mm[k0][:, boot].mean(2), mm[n][:, boot].mean(2))   # (nb, n_boot, C)
        se = sb.std(1, ddof=1)
        thr = np.maximum(fixed_floor, 2 * se)
        ok = (stat < thr) & (np.abs(slope) < fixed_slope)
        return ok, slope, stat, se, thr

    T = t[-1]
    fixed_now, slope_half, stat, stat_se, thr = fixed_at(N - 1)
    if N - 1 - H >= 1 and t[N - 1 - H] > 0:
        fixed_prev = fixed_at(N - 1 - H)[0]
    else:
        fixed_prev = np.zeros_like(fixed_now)
    fixed = fixed_now & fixed_prev

    t_half = np.array([[_first_crossing(t, Lm[:, b, c], 0.5 * Lm[0, b, c]) for c in range(C)]
                       for b in range(nb)])
    n, dec = N - 1, PER_DECADE
    s1 = _index_slope(t, logL, n - dec, n)
    s2 = _index_slope(t, logL, n - 2 * dec, n - dec)
    s1_se = _index_slope(t, logLb, n - dec, n).std(1, ddof=1)
    tail = (np.nan_to_num(t_half, nan=np.inf) <= T / 10) & (np.abs(s1 - s2) < tail_dslope) \
        & (s1_se < tail_se)

    status = np.where(fixed, "FIXED", np.where(tail, "TAIL",
                                                "UNRESOLVED" if at_cap else "MOVING"))
    out = {"T": float(T), "status": status, "fixed": fixed, "fixed_now": fixed_now,
           "fixed_prev": fixed_prev, "tail": tail, "t_half": t_half,
           "slope_half_decade": slope_half, "slope_last_decade": s1, "slope_prev_decade": s2,
           "slope_last_se": s1_se, "stationarity": stat, "stationarity_se": stat_se,
           "stationarity_threshold": thr, "loss_last": Lm[-1]}
    if "frac_eps" in rec:
        out["frac_eps_last"] = np.asarray(rec["frac_eps"], float)[-1].mean(1)
    out["counts"] = {s: int((status == s).sum()) for s in ("FIXED", "TAIL", "MOVING",
                                                           "UNRESOLVED")}
    out["certified"] = bool(np.isin(status, ("FIXED", "TAIL")).all())
    return out


# --------------------------------------------------------------------------- #
# GPU part (JAX imported lazily)
# --------------------------------------------------------------------------- #
@functools.lru_cache(maxsize=None)
def jax_modules():
    zs, jax, jnp, discounts, dynamics = I.jax_modules()
    jax.config.update("jax_compilation_cache_dir", str(OUT / "jaxcache"))
    import problems
    import simulate
    return zs, jax, jnp, problems, simulate


def _dtype(jnp, dtype):
    return {"f64": jnp.float64, "f32": jnp.float32}[dtype]


def _build_sim(opt, priors, factor, means, lr, beta2, eps):
    """Simulator of one lane from traced ``(priors, factor, means, lr, beta2, eps)`` (no
    bias, batch 1, step ``lr / D``)."""
    zs, jax, jnp, problems, simulate = jax_modules()
    D = means.shape[0]
    data = simulate.MixtureData(factor, means, jnp.zeros((D, 0), means.dtype))
    problem = problems.make_gmm_softmax(priors, risk_samples=1)
    if opt == "adam":
        sim = simulate.build_mixture_adam(problem, data, lr / D, beta1=BETA1, beta2=beta2,
                                          eps=eps, batch=BATCH)
    else:
        sim = simulate.build_mixture_sgd(problem, data, lr / D, batch=BATCH)
    return sim, data


def _observe(opt, factor, means, eps, params, state):
    """Per-lane observables after a segment (see the module docstring)."""
    zs, jax, jnp, problems, simulate = jax_modules()
    K = factor ** 2
    mm = functools.partial(jnp.matmul, precision=jax.lax.Precision.HIGHEST)
    B = mm(params.T, K[:, None] * params)
    O = mm(params.T, means)
    C = params.shape[1]
    if opt == "adam":
        m, v = state
        stab = eps * K[:, None]
        den = jnp.sqrt(v + stab)
        u = m / den
        vbar = v.mean(0)
        nu = (jnp.abs(m) / den).mean(0)
        frac = (v < stab).astype(params.dtype).mean(0)
        U = mm(means.T, u)
    else:
        nan = jnp.full((C,), jnp.nan, params.dtype)
        vbar = nu = frac = nan
        U = jnp.full((C, C), jnp.nan, params.dtype)
    return {"B": B, "O": O, "U": U, "vbar": vbar, "nu": nu, "frac_eps": frac}


@functools.lru_cache(maxsize=None)
def _chunk_fn():
    zs, jax, jnp, problems, simulate = jax_modules()

    @functools.partial(jax.jit, static_argnames=("opt", "unroll"))
    def chunk(carry, bounds, beta2, priors, factor, means, lr, eps, *, opt, unroll):
        """Advance all lanes over the ``(H, 2)`` ``bounds``; ``carry = (params, state, key)``
        with a leading lane axis, ``beta2`` ``(lanes,)``."""
        def lane(c, b2):
            sim, data = _build_sim(opt, priors, factor, means, lr, b2, eps)
            obs = functools.partial(_observe, opt, factor, means, eps)
            return simulate.run_mixture_segments(sim, data, c, bounds, observe=obs,
                                                 unroll=unroll)
        return jax.vmap(lane)(carry, beta2)

    return chunk


def class_metrics(B, O, *, samples=SAMPLES, seed=0, root=None, draws=None, err_batch=32):
    """Per-class ``(loss, err, loss_cv)`` for one ``(B, O)`` on the fixed draws of
    ``problems.softmax_class_metrics`` (``noise = draws @ sym_sqrt(B)``, ``draws =
    problems._gaussian_draws(C, samples, seed)``), float64.

    * ``loss``: the same estimator ``mean_s [logsumexp_k q_sk - q_sc]``, computed without
      cancellation as ``mean_s softplus(log Y_sc)``, ``Y_sc = sum_{k != c} exp(q_sk -
      q_sc)``, the sums formed by one ``(S x C) @ (C x C)`` matmul of max-shifted
      exponentials (selftest 6/8: equal to ``softmax_class_metrics`` to round-off).
    * ``err``: ``mean_s [argmax_k q_sk != c]`` exactly as ``softmax_class_metrics``.
    * ``loss_cv``: regression control variate ``Y`` with exact mean ``sum_{k != c}
      exp(O_kc - O_cc + s_kc^2/2)``, ``s_kc^2 = B_kk + B_cc - 2 B_kc``: near-exact in the
      deep tail, where plain MC with ``2^14`` draws has several-percent relative error
      (selftest 8). Used only where the ``EY``-weighted ``s^2`` is below ``CV_MAX_VAR = 6``
      (beyond it ``E Y`` sits in lognormal tails the draws never reach and the control
      variate is badly biased); elsewhere ``loss_cv = loss``. Calibrated on cached
      series-I C = 128 states against 2^20 draws: max relative error 2.1% (plain 6.4%)
      for ``s^2 < 6`` (0.35% for ``s^2 < 2``); at ``s^2 ~ 15-26`` it was off by 20-90%.

    ``root`` (``sym_sqrt(B)``) and ``draws`` may be precomputed (batched ``eigh`` is ~60x
    faster on the GPU than one-by-one)."""
    zs, jax, jnp, problems, simulate = jax_modules()
    from utils import sym_sqrt
    C = B.shape[0]
    if draws is None:
        draws = problems._gaussian_draws(C, samples, seed)
    if root is None:
        root = sym_sqrt(B)
    G = draws @ root                                           # (S, C) logit noise
    a = G.max(1, keepdims=True)
    b = O.max(0, keepdims=True)
    off = 1.0 - jnp.eye(C, dtype=O.dtype)
    num = jnp.exp(G - a) @ (jnp.exp(O - b) * off)            # sum_{k != c}, shifted
    logY = jnp.log(num) - (G - a) - (jnp.diagonal(O) - b[0])[None, :]
    X, Y = jax.nn.softplus(logY), jnp.exp(logY)
    dB = jnp.diagonal(B)
    var = dB[:, None] + dB[None, :] - 2 * B                  # Var(q_k - q_c), [k, c]
    logw = jnp.where(off > 0, O - jnp.diagonal(O)[None, :] + var / 2, -jnp.inf)
    EY = jnp.sum(jnp.exp(logw), 0)
    s_eff = jnp.sum(jax.nn.softmax(logw, axis=0) * var, 0)   # EY-weighted variance
    Xm, Ym = X.mean(0), Y.mean(0)
    beta = jnp.mean((X - Xm) * (Y - Ym), 0) / jnp.maximum(jnp.mean((Y - Ym) ** 2, 0), 1e-300)

    def one(c):
        q = O[:, c][None, :] + G
        return jnp.mean((jnp.argmax(q, axis=1) != c).astype(q.dtype))

    err = jax.lax.map(one, jnp.arange(C), batch_size=err_batch)
    return Xm, err, jnp.where(s_eff < CV_MAX_VAR, Xm - beta * (Ym - EY), Xm)


@functools.lru_cache(maxsize=None)
def _metrics_fn():
    zs, jax, jnp, problems, simulate = jax_modules()

    @functools.partial(jax.jit, static_argnames=("nb", "runs"))
    def metrics(rec, draws, *, nb, runs):
        """Reduce lane records ``(lanes, H, ...)`` to per-run ``(H, nb, runs, C)`` and seed
        means ``(H, nb, C, C)``; losses in float64."""
        f64 = jnp.float64
        B = jnp.swapaxes(rec["B"], 0, 1).astype(f64)        # (H, L, C, C)
        O = jnp.swapaxes(rec["O"], 0, 1).astype(f64)
        U = jnp.swapaxes(rec["U"], 0, 1)
        Hn, L, C = B.shape[0], B.shape[1], B.shape[-1]
        from utils import sym_sqrt
        flatB = B.reshape(-1, C, C)
        flat = (flatB, O.reshape(-1, C, C), jax.vmap(sym_sqrt)(flatB))
        loss, err, loss_cv = jax.lax.map(
            lambda bor: class_metrics(bor[0], bor[1], root=bor[2], draws=draws), flat)
        per = lambda x: x.reshape((Hn, nb, runs) + x.shape[2:])
        diag = lambda x: jnp.diagonal(x, axis1=-2, axis2=-1)
        Od = diag(O)
        eye = jnp.eye(C, dtype=bool)
        mean_off = (O.sum(-2) - Od) / (C - 1)
        max_off = jnp.where(eye, -jnp.inf, O).max(-2)
        out = {"loss": per(loss.reshape(Hn, L, C)), "err": per(err.reshape(Hn, L, C)),
               "loss_cv": per(loss_cv.reshape(Hn, L, C)),
               "Bdiag": per(diag(B)), "Odiag": per(Od), "margin_mean": per(Od - mean_off),
               "margin_min": per(Od - max_off)}
        for k in ("vbar", "nu", "frac_eps"):
            out[k] = per(jnp.swapaxes(rec[k], 0, 1))
        out["Udiag"] = per(diag(U))
        out["B_mean"] = per(B).mean(2)
        out["O_mean"] = per(O).mean(2)
        out["U_mean"] = per(U).mean(2).astype(jnp.float32)
        return out

    return metrics


def numerics_fingerprint():
    """sha256 of everything that determines a trajectory: ``simulate.py``, ``problems.py``,
    ``utils.py``, the knee functions that build the executables, and the JAX version."""
    import jax
    h = hashlib.sha256()
    for f in NUMERICS_FILES:
        h.update((ROOT / f).read_bytes())
    for fn in (_build_sim, _observe, _chunk_fn, _metrics_fn, class_metrics, record_steps,
               chunk_bounds, init_carry, model_arrays, run_keys, grid_index, lane_beta2,
               Runner, _priors):
        h.update(inspect.getsource(fn).encode())
    h.update(repr((H, PER_DECADE, T_MIN, K_OFFSET, SAMPLES, CV_MAX_VAR, BETA1, BATCH, R)).encode())
    h.update(jax.__version__.encode())
    return h.hexdigest()[:16]


def run_keys(runs, seed=SEED_J):
    """``key_r = fold_in(PRNGKey(seed), r)``, ``r = 0 .. runs - 1`` (prefix-stable)."""
    zs, jax, jnp, problems, simulate = jax_modules()
    base = jax.random.PRNGKey(seed)
    return jax.vmap(lambda r: jax.random.fold_in(base, r))(jnp.arange(runs))


def model_arrays(g: Group):
    """``(priors, factor, means, theta0)`` in the group dtype (priors float64)."""
    zs, jax, jnp, problems, simulate = jax_modules()
    priors, data, theta0 = zs.zipf_model(g.classes, g.alpha, g.dimension, r=R,
                                         noise=g.kappa)
    dt = _dtype(jnp, g.dtype)
    return (jnp.asarray(priors, jnp.float64), data.factor.astype(dt), data.means.astype(dt),
            theta0.astype(dt))


def init_carry(g: Group, theta0):
    """Lane carry ``(params, state, key)``; lane ``b * runs + r`` has ``beta2_b``, ``key_r``."""
    zs, jax, jnp, problems, simulate = jax_modules()
    L = g.lanes
    params = jnp.broadcast_to(theta0, (L,) + theta0.shape)
    state = (jnp.zeros_like(params), jnp.zeros_like(params)) if g.opt == "adam" else ()
    keys = jnp.tile(run_keys(g.runs, g.seed), (len(g.lane_beta2), 1))
    return params, state, keys


def lane_beta2(g: Group):
    zs, jax, jnp, problems, simulate = jax_modules()
    b = np.repeat(np.nan_to_num(np.asarray(g.lane_beta2, float), nan=0.0), g.runs)
    return jnp.asarray(b, _dtype(jnp, g.dtype))


class Runner:
    """Arrays and executables of one group; :meth:`step` runs chunk ``n`` from ``carry``."""

    def __init__(self, g: Group, samples=SAMPLES):
        zs, jax, jnp, problems, simulate = jax_modules()
        self.g, self.samples = g, samples
        self.priors, self.factor, self.means, self.theta0 = model_arrays(g)
        dt = _dtype(jnp, g.dtype)
        self.beta2 = lane_beta2(g)
        self.lr = jnp.asarray(g.lr, dt)
        self.eps = jnp.asarray(g.eps, dt)
        self.chunk = _chunk_fn()
        self.metrics = _metrics_fn()
        self.draws = problems._gaussian_draws(g.classes, samples, 0)

    def init(self):
        return init_carry(self.g, self.theta0)

    def advance(self, carry, n):
        zs, jax, jnp, problems, simulate = jax_modules()
        bounds = jnp.asarray(chunk_bounds(self.g.dimension, n))
        return self.chunk(carry, bounds, self.beta2, self.priors, self.factor, self.means,
                          self.lr, self.eps, opt=self.g.opt, unroll=self.g.unroll)

    def reduce(self, rec):
        return self.metrics(rec, self.draws, nb=len(self.g.lane_beta2), runs=self.g.runs)


# --------------------------------------------------------------------------- #
# Storage: chunks, checkpoints, meta
# --------------------------------------------------------------------------- #
def group_dir(g_or_name, root=OUT):
    return Path(root) / "runs" / (g_or_name if isinstance(g_or_name, str) else name(g_or_name))


def _atomic_write_text(path, text):
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def _atomic_savez(path, **arrays):
    tmp = path.with_name(path.name + ".tmp.npz")
    np.savez(tmp, **arrays)
    os.replace(tmp, path)


def pinned(g: Group, samples=SAMPLES):
    return {"name": name(g), "opt": g.opt, "classes": g.classes, "alpha": g.alpha,
            "kappa": g.kappa, "eps": g.eps if g.opt == "adam" else None, "dimension": g.dimension, "runs": g.runs,
            "seed": g.seed, "dtype": g.dtype, "unroll": g.unroll, "lr": g.lr,
            "beta2s": [float(b) for b in g.lane_beta2] if g.opt == "adam" else [],
            "H": H, "lanes": g.lanes, "per_decade": PER_DECADE, "t_min": T_MIN,
            "k_offset": K_OFFSET,
            "samples": samples, "beta1": BETA1, "r": R, "batch": BATCH}


def read_meta(d):
    p = Path(d) / "meta.json"
    return json.loads(p.read_text()) if p.exists() else None


def save_checkpoint(d, carry, n_next, meta):
    """Checkpoint after chunk ``n_next - 1``: ``ckpt/{n_next:03d}/*.npy`` then an atomic
    ``meta.json`` pointing at it; older checkpoint directories are removed afterwards."""
    params, state, key = carry
    ck = Path(d) / "ckpt" / f"{n_next:03d}"
    if ck.exists():
        shutil.rmtree(ck)
    ck.mkdir(parents=True)
    np.save(ck / "params.npy", np.asarray(params))
    if len(state):
        np.save(ck / "m.npy", np.asarray(state[0]))
        np.save(ck / "v.npy", np.asarray(state[1]))
    np.save(ck / "key.npy", np.asarray(key))
    step = int(chunk_bounds(meta["dimension"], n_next - 1)[-1, 1])
    np.save(ck / "step.npy", np.array(step, np.int64))
    np.save(ck / "record_index.npy", np.array(n_next * H, np.int64))
    meta = dict(meta, next_chunk=n_next, step=step, record_index=n_next * H,
                time=step / meta["dimension"], ckpt=f"ckpt/{n_next:03d}")
    _atomic_write_text(Path(d) / "meta.json", json.dumps(meta, indent=1))
    for old in (Path(d) / "ckpt").iterdir():
        if old.name != f"{n_next:03d}":
            shutil.rmtree(old, ignore_errors=True)
    return meta


def load_checkpoint(d, meta, dtype):
    zs, jax, jnp, problems, simulate = jax_modules()
    ck = Path(d) / meta["ckpt"]
    params = jnp.asarray(np.load(ck / "params.npy"))
    state = ((jnp.asarray(np.load(ck / "m.npy")), jnp.asarray(np.load(ck / "v.npy")))
             if (ck / "m.npy").exists() else ())
    key = jnp.asarray(np.load(ck / "key.npy"))
    assert int(np.load(ck / "record_index.npy")) == meta["next_chunk"] * H
    return params, state, key


def check_meta(meta, g, samples, allow_code_change=False):
    """Refuse to extend a group whose pinned configuration or numerics differ."""
    want = pinned(g, samples)
    bad = {k: (meta.get(k), v) for k, v in want.items() if meta.get(k) != v}
    if bad:
        raise SystemExit(f"{name(g)}: checkpoint configuration mismatch {bad}; refusing to extend")
    fp, gh = numerics_fingerprint(), git_hash()
    if meta.get("fingerprint") != fp:
        msg = (f"{name(g)}: numerics fingerprint {meta.get('fingerprint')} (git "
               f"{meta.get('git')}) != current {fp} (git {gh})")
        if not allow_code_change:
            raise SystemExit(msg + "; refusing to extend (--allow-code-change overrides)")
        print("WARNING: " + msg + " (allowed)", flush=True)
    dev = device_info()
    if any(meta.get(k) not in (None, v) for k, v in dev.items()):
        msg = (f"{name(g)}: checkpoint written on {meta.get('device')} / jaxlib "
               f"{meta.get('jaxlib')}, now {dev['device']} / {dev['jaxlib']}")
        if not allow_code_change:
            raise SystemExit(msg + "; refusing to extend (--allow-code-change overrides)")
        print("WARNING: " + msg + " (allowed)", flush=True)
    if meta.get("fingerprint") == fp and meta.get("git") != gh:
        print(f"note: {name(g)}: git {meta.get('git')} -> {gh}, numerics fingerprint unchanged",
              flush=True)


def chunk_path(d, n):
    return Path(d) / f"chunk_{n:03d}.npz"


def load_group(g_or_dir, root=OUT, upto=None):
    """Concatenate ``chunk_*.npz`` of a group along the record axis. Returns a dict with
    ``time, steps, record`` ``(N,)``, per-run ``(N, nb, runs, C)``, seed means
    ``(N, nb, C, C)``, and ``priors, beta2s, meta``."""
    d = Path(g_or_dir) if isinstance(g_or_dir, (str, Path)) and Path(g_or_dir).is_dir() \
        else group_dir(g_or_dir, root)
    meta = read_meta(d)
    n_done = meta["next_chunk"] if meta else len(list(d.glob("chunk_*.npz")))
    if upto is not None:
        n_done = min(n_done, upto)
    parts = [dict(np.load(chunk_path(d, n))) for n in range(n_done)]
    if not parts:
        return None
    static = ("priors", "beta2s", "chunk", "seconds_sim", "seconds_metrics")
    out = {k: np.concatenate([p[k] for p in parts]) for k in parts[0] if k not in static}
    out["priors"], out["beta2s"] = parts[0]["priors"], parts[0]["beta2s"]
    out["meta"] = meta
    return out


# --------------------------------------------------------------------------- #
# Running a group
# --------------------------------------------------------------------------- #
def log_line(root, g, text):
    p = Path(root) / "logs"
    p.mkdir(parents=True, exist_ok=True)
    with open(p / f"{name(g)}.log", "a") as fh:
        fh.write(text + "\n")


def append_jsonl(path, record):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as fh:
        fh.write(json.dumps(record) + "\n")


def run_chunk(runner, carry, n, d):
    """Run chunk ``n``, reduce and store it; returns ``(carry, seconds_sim, seconds_metrics)``."""
    zs, jax, jnp, problems, simulate = jax_modules()
    g = runner.g
    t0 = time.time()
    carry, rec = runner.advance(carry, n)
    jax.block_until_ready(carry)
    t1 = time.time()
    red = runner.reduce(rec)
    red = {k: np.asarray(v) for k, v in red.items()}
    t2 = time.time()
    steps = record_steps(g.dimension, (n + 1) * H)[chunk_records(n)]
    _atomic_savez(chunk_path(d, n), record=chunk_records(n), steps=steps,
                  time=steps / g.dimension, priors=np.asarray(runner.priors),
                  beta2s=np.asarray(g.lane_beta2, float), chunk=np.array(n),
                  seconds_sim=np.array(t1 - t0), seconds_metrics=np.array(t2 - t1), **red)
    return carry, t1 - t0, t2 - t1


def open_group(g, root=OUT, samples=SAMPLES, allow_code_change=False):
    """``(runner, carry, meta)``: resumes from the checkpoint or starts fresh."""
    d = group_dir(g, root)
    d.mkdir(parents=True, exist_ok=True)
    runner = Runner(g, samples)
    meta = read_meta(d)
    if meta is None:
        meta = dict(pinned(g, samples), fingerprint=numerics_fingerprint(), git=git_hash(),
                    **device_info(),
                    git_dirty=git_dirty(), next_chunk=0, step=0, record_index=0, time=0.0,
                    ckpt=None, history=[])
        _atomic_write_text(d / "meta.json", json.dumps(meta, indent=1))
        return runner, runner.init(), meta
    check_meta(meta, g, samples, allow_code_change)
    if meta["next_chunk"] == 0:
        return runner, runner.init(), meta
    return runner, load_checkpoint(d, meta, g.dtype), meta


def advance_group(g, n_chunks, *, root=OUT, samples=SAMPLES, allow_code_change=False,
                  wall_deadline=None, quiet=False, opened=None):
    """Run ``g`` until ``n_chunks`` chunks exist (checkpoint after every chunk)."""
    runner, carry, meta = opened or open_group(g, root, samples, allow_code_change)
    d = group_dir(g, root)
    n0, t_start = meta["next_chunk"], time.time()
    sim_s = met_s = 0.0
    for n in range(meta["next_chunk"], n_chunks):
        if wall_deadline is not None and time.time() > wall_deadline:
            break
        carry, a, b = run_chunk(runner, carry, n, d)
        sim_s, met_s = sim_s + a, met_s + b
        meta = save_checkpoint(d, carry, n + 1, meta)
        line = (f"chunk {n:03d}: T={chunk_time(g.dimension, n):.4g} step={meta['step']}"
                f" sim {a:.2f}s metrics {b:.2f}s")
        log_line(root, g, line)
        if not quiet:
            print(f"{name(g)} {line}", flush=True)
    if meta["next_chunk"] > n0:
        wall = time.time() - t_start
        meta["history"] = meta.get("history", []) + [
            {"chunks": [n0, meta["next_chunk"]], "git": git_hash(),
             "fingerprint": numerics_fingerprint(), "wall": wall, "unix": time.time()}]
        _atomic_write_text(d / "meta.json", json.dumps(meta, indent=1))
        append_jsonl(Path(root) / "runs.jsonl",
                     {"group": name(g), "chunks": [n0, meta["next_chunk"]],
                      "steps": meta["step"], "T": meta["time"], "wall": wall,
                      "sim_seconds": sim_s, "metrics_seconds": met_s, "lanes": g.lanes,
                      "git": git_hash(), "fingerprint": numerics_fingerprint()})
    return runner, carry, meta


def run_group(g, T0, *, cap, extend=False, root=OUT, samples=SAMPLES,
              allow_code_change=False, wall_hours=None, quiet=False):
    """Run to ``T0``; with ``extend``, add one chunk (half a decade) at a time until
    every class of every lane is certified (FIXED/TAIL) or ``cap`` / the wall cap."""
    deadline = time.time() + 3600 * wall_hours if wall_hours else None
    target = chunks_for(T0)
    opened = open_group(g, root, samples, allow_code_change)
    while True:
        opened = advance_group(g, target, root=root, samples=samples, wall_deadline=deadline,
                               quiet=quiet, opened=opened)
        meta = opened[2]
        rec = load_group(g, root)
        T_now = meta["time"]
        T_next = chunk_time(g.dimension, meta["next_chunk"])   # end of the next chunk
        at_cap = (T_now >= cap * (1 - 1e-9) or T_next > cap * (1 + 1e-9)
                  or (deadline is not None and time.time() > deadline)
                  or meta["next_chunk"] < target)
        cert = certify(rec, at_cap=at_cap)
        summary = {"group": name(g), "T": T_now, "steps": meta["step"],
                   "chunks": meta["next_chunk"], "certified": cert["certified"],
                   "at_cap": bool(at_cap), "cap": cap, "unix": time.time(),
                   "counts_by_beta2": [{s: int((st == s).sum()) for s in
                                        ("FIXED", "TAIL", "MOVING", "UNRESOLVED")}
                                       for st in cert["status"]],
                   "beta2s": [float(b) for b in g.lane_beta2] if g.opt == "adam" else []}
        append_jsonl(Path(root) / "status.jsonl", summary)
        if not quiet:
            print(f"{name(g)}: T={T_now:.4g} certified={cert['certified']} "
                  f"{summary['counts_by_beta2']}", flush=True)
        if not extend or cert["certified"] or at_cap:
            return cert
        target = meta["next_chunk"] + 1


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def groups_from_args(args):
    out = []
    for C in args.classes:
        for alpha in args.alphas:
            for kappa in args.kappas:
                dims = args.dimension or [default_dimension(kappa)]
                for D in dims:
                    for opt in args.opts:
                        for eps in (args.eps if opt == "adam" else [0.0]):
                            for dtype in args.dtype:
                                for U in args.unroll:
                                    eta = args.eta_adam if opt == "adam" else args.eta_sgd
                                    out.append(Group(opt, C, float(alpha), float(kappa),
                                                     float(eps), int(D), args.runs,
                                                     args.seed, dtype, int(U), eta,
                                                     tuple(args.beta2s)))
    return out


def cmd_run(args):
    for g in groups_from_args(args):
        T0, t_char = horizon_rule(g.opt, g.classes, g.alpha, g.kappa, g.beta2s)
        if args.horizon is not None:
            T0 = args.horizon
        cap = args.cap if args.cap is not None else default_cap(t_char, g.dimension, T0)
        n = chunks_for(T0)
        T_end = chunk_time(g.dimension, n - 1)
        meta = read_meta(group_dir(g, args.root))
        done = meta["next_chunk"] if meta else 0
        mem = g.lanes * g.dimension * g.classes * (8 if g.dtype == "f64" else 4) * \
            (3 if g.opt == "adam" else 1) / 2 ** 30
        print(f"{name(g)}: T0={T0:.4g} -> {n} chunks (T={T_end:.4g}, "
              f"{int(T_end * g.dimension)} steps), cap T={cap:.4g}, done {done} chunks,"
              f" lanes {g.lanes}, carry {mem:.2f} GiB", flush=True)
        if args.dry_run:
            continue
        run_group(g, T0, cap=cap, extend=args.extend_until_certified, samples=args.samples,
                  root=args.root,
                  allow_code_change=args.allow_code_change, wall_hours=args.wall_hours)


def cmd_status(args):
    root = Path(args.root)
    runs = root / "runs"
    last = {}
    if (root / "status.jsonl").exists():
        for line in (root / "status.jsonl").read_text().splitlines():
            r = json.loads(line)
            last[r["group"]] = r
    if not runs.exists():
        print("no groups yet")
        return
    for d in sorted(runs.iterdir()):
        meta = read_meta(d)
        if meta is None:
            continue
        size = sum(f.stat().st_size for f in d.rglob("*") if f.is_file()) / 2 ** 20
        wall = sum(h["wall"] for h in meta.get("history", []))
        s = last.get(d.name)
        cert = (f"certified={s['certified']} at_cap={s['at_cap']} {s['counts_by_beta2']}"
                if s else "no certificate yet")
        print(f"{d.name}: chunks {meta['next_chunk']} T={meta['time']:.4g} step={meta['step']}"
              f" wall {wall:.0f}s {size:.1f} MiB git {meta['git']} | {cert}")


def cmd_probe(args):
    """Seconds per step (all lanes) of the chunk executable, compile and metrics times."""
    zs, jax, jnp, problems, simulate = jax_modules()
    results = []
    probe_file = Path(args.root) / "probe.json"
    C = args.classes[0]
    for dtype in args.dtype:
        for D in (args.dimension or [1000]):
            for L in args.lanes:
                for U in args.unroll:
                    for opt in args.opts:
                        nb = len(args.beta2s) if opt == "adam" else 1
                        runs = max(1, L // nb)
                        g = Group(opt, C, float(args.alphas[0]), float(args.kappas[0]),
                                  float(args.eps[0]), int(D), runs, args.seed, dtype, U,
                                  None, tuple(args.beta2s))
                        runner = Runner(g, args.samples)
                        carry = runner.init()
                        n_steps = args.probe_steps
                        per = n_steps // H
                        b = np.stack([np.arange(H) * per, (np.arange(H) + 1) * per], 1)
                        bounds = jnp.asarray(b.astype(np.int64))
                        dyn = (carry, bounds, runner.beta2, runner.priors, runner.factor,
                               runner.means, runner.lr, runner.eps)
                        t0 = time.time()
                        compiled = runner.chunk.lower(*dyn, opt=opt, unroll=U).compile()
                        t_comp = time.time() - t0
                        out = compiled(*dyn)
                        jax.block_until_ready(out)                  # warm-up
                        reps = []
                        for _ in range(args.probe_reps):
                            t0 = time.time()
                            out = compiled(*dyn)
                            jax.block_until_ready(out)
                            reps.append(time.time() - t0)
                        rec = out[1]
                        t0 = time.time()
                        red = runner.reduce(rec)
                        jax.block_until_ready(red)
                        t_mcomp = time.time() - t0
                        t0 = time.time()
                        red = runner.reduce(rec)
                        jax.block_until_ready(red)
                        t_met = time.time() - t0
                        best = min(reps)
                        r = {"opt": opt, "C": C, "D": int(D), "lanes": g.lanes, "unroll": U,
                             "dtype": dtype, "steps": int(per * H),
                             "us_per_step": 1e6 * best / (per * H),
                             "us_per_lane_step": 1e6 * best / (per * H) / g.lanes,
                             "compile_s": t_comp, "metrics_first_s": t_mcomp,
                             "metrics_per_chunk_s": t_met, "samples": args.samples,
                             "reps": reps, "git": git_hash(), "unix": time.time(),
                             "device": str(jax.devices()[0])}
                        results.append(r)
                        print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v)
                                          for k, v in r.items() if k != "reps"}), flush=True)
    probe_file.parent.mkdir(parents=True, exist_ok=True)
    old = json.loads(probe_file.read_text()) if probe_file.exists() else []
    _atomic_write_text(probe_file, json.dumps(old + results, indent=1))


def cmd_selftest(args):
    sys.argv = [str(HERE / "knee" / "selftest.py")] + (["--quick"] if args.quick else [])
    runpy.run_path(str(HERE / "knee" / "selftest.py"), run_name="__main__")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=["selftest", "probe", "run", "status"])
    p.add_argument("--classes", nargs="+", type=int, default=[128])
    p.add_argument("--alphas", nargs="+", type=float, default=[0.0])
    p.add_argument("--kappas", nargs="+", type=float, default=[1.0])
    p.add_argument("--eps", nargs="+", type=float, default=[1e-8])
    p.add_argument("--opts", nargs="+", default=["adam", "sgd"], choices=["adam", "sgd"])
    p.add_argument("--beta2s", nargs="+", type=float, default=list(BETA2S))
    p.add_argument("--runs", type=int, default=32)
    p.add_argument("--seed", type=int, default=SEED_J)
    p.add_argument("--dtype", nargs="+", default=["f64"], choices=["f64", "f32"])
    p.add_argument("--unroll", nargs="+", type=int, default=[1])
    p.add_argument("--dimension", nargs="+", type=int, default=None,
                   help="D (default max(1000, ceil(20*25/kappa)); probe default 1000)")
    p.add_argument("--horizon", type=float, default=None,
                   help="initial T in time units (default: the plan's horizon rule)")
    p.add_argument("--extend-until-certified", action="store_true")
    p.add_argument("--cap", type=float, default=None,
                   help="T cap (time units; default min(100 t_A, 2e8/D))")
    p.add_argument("--wall-hours", type=float, default=None)
    p.add_argument("--eta-adam", type=float, default=None)
    p.add_argument("--eta-sgd", type=float, default=None)
    p.add_argument("--samples", type=int, default=SAMPLES)
    p.add_argument("--allow-code-change", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--root", type=Path, default=OUT,
                   help="output root (default figs/experiments/knee; jaxcache stays there)")
    # probe
    p.add_argument("--lanes", nargs="+", type=int, default=[8, 32, 128],
                   help="probe: total lanes (Adam: runs = lanes / n_beta2)")
    p.add_argument("--probe-steps", type=int, default=4000)
    p.add_argument("--probe-reps", type=int, default=2)
    # selftest
    p.add_argument("--quick", action="store_true")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    {"selftest": cmd_selftest, "probe": cmd_probe, "run": cmd_run,
     "status": cmd_status}[args.command](args)


if __name__ == "__main__":
    main()
