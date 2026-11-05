"""Helpers for the series-I analysis (numpy/scipy only; reads figs/experiments/imbalance).

Reuses ``experiments/zipf/analysis/e1_timescales/e1b_lib.py`` (ols, bpl_sharp,
bpl_smooth, local_slopes, first_crossing, interp_log) and carries parameterised copies
of the E1a smoothly broken power law (``e1_rates/analysis.py``) and of the
background/eps prefactor ``F`` (``e1_rates/analysis2.py``).
"""
from __future__ import annotations

import functools
import os
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
DATA = REPO / "figs" / "experiments" / "imbalance"
OUTDIR = DATA / "analysis"
OUTDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", "/tmp/adam-matplotlib")

sys.path.insert(0, str(REPO / "experiments" / "zipf" / "analysis" / "e1_timescales"))
from e1b_lib import bpl_sharp, bpl_smooth, first_crossing, interp_log, local_slopes, ols  # noqa

sys.path.insert(0, str(REPO / "experiments"))
import imbalance as I  # noqa: E402  (numpy-only at import)

R, BETA1, EPS, BATCH = I.R, I.BETA1, I.EPS, I.BATCH
ETA = I.ETA
LEVELS = (0.75, 0.5, 0.25)


# --------------------------------------------------------------------------- #
# Theory
# --------------------------------------------------------------------------- #
def kappa_beta(beta2, beta1=BETA1):
    return (np.pi / 2) * ((1 - beta1 / np.sqrt(beta2)) / (1 - beta1)) ** 2


def coef_beta(beta2, beta1=BETA1):
    """Rare-event rate / (r^2 b pi): ``c_beta sqrt(2/pi) / (1 - beta1/sqrt(beta2))``."""
    c_beta = (1 - beta1) / np.sqrt(1 - beta2)
    return c_beta * np.sqrt(2 / np.pi) / (1 - beta1 / np.sqrt(beta2))


_XI = np.linspace(-9, 9, 360001)
_W_XI = np.exp(-_XI ** 2 / 2) / np.sqrt(2 * np.pi) * (_XI[1] - _XI[0])


def g(a):
    a = np.atleast_1d(a)
    return np.array([np.sum(_W_XI * _XI ** 2 / np.sqrt(_XI ** 2 + aa)) for aa in a])


@functools.lru_cache(maxsize=None)
def F_pred(beta2, C, b=BATCH, eps=EPS, beta1=BETA1, tol=1e-8):
    """Rare-side prefactor (``analysis2.F_pred``) with ``C`` explicit and lags summed
    until ``rho^l < tol`` (88 lags bias it by ~1% at ``beta2 = 0.9``)."""
    rho = beta1 / np.sqrt(beta2)
    f, pc = 1 - 1 / C, 1 / C
    a = b * b * (eps + pc ** 2 / b) / (1 - beta2)
    lags = int(np.ceil(np.log(tol) / np.log(rho)))
    l = np.arange(lags)
    al = a / (f ** 2 * beta2 ** l)
    # g is smooth in log a: evaluate on a coarse log grid and interpolate
    grid = np.geomspace(al.min(), al.max() * 1.0001, 200)
    gl = np.interp(np.log(al), np.log(grid), g(grid))
    return float(np.sum(rho ** l * gl) / (np.sqrt(2 / np.pi) * np.sum(rho ** l)))


def kink(beta2, C=None, b=BATCH):
    """``kappa_beta (1 - beta2) / (b F^2)`` (``C`` given) or without ``F``."""
    base = kappa_beta(beta2) * (1 - beta2) / b
    return base if C is None else base / F_pred(beta2, C) ** 2


def ratio_asymptotes(beta2, C, kappa, etaset="base"):
    """Predicted ``t_c(Adam)/t_c(SGD)`` at ``Theta_0``: ``(window(pi), rare)`` with
    ``window = 10 sqrt(pi)(1 - 1/C) sqrt(kappa)`` and ``rare = 10 (1 - 1/C) sqrt(kappa) /
    (coef_beta F)``; they meet at :func:`kink`. ``sqrt(kappa)``: Adam's frozen rate
    scales by ``r'^2 eta'/eta = r^2/sqrt(kappa)``, SGD's by ``r^2``."""
    fac = (I.eta_of("sgd", kappa, etaset) / I.eta_of("adam", kappa, etaset)
           * (1 - 1 / C) * np.sqrt(kappa))
    return (lambda pi: fac * np.sqrt(pi)), fac / (coef_beta(beta2) * F_pred(beta2, C))


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def run(model, opt, kappa, C, alpha, beta2=0.0, etaset="base"):
    T = I.ode_horizon(opt, C)
    return I.Run(model, opt, kappa, C, alpha, beta2 if opt == "adam" else 0.0, T,
                 etaset=etaset)


@functools.lru_cache(maxsize=None)
def _ode_file(model, opt, kappa, C, alpha, beta2, etaset, safety):
    """Longest cached ODE trajectory with these parameters (runs are extended in place)."""
    r = run(model, opt, kappa, C, alpha, beta2, etaset)._replace(horizon=1.0, safety=safety)
    pattern = I.stem(r).replace("_T1_", "_T*_") + ".npz"
    best, best_T = None, -1.0
    for f in (DATA / "ode").glob(pattern):
        T = float(f.name.split("_T")[-1].split("_s")[0])
        if T > best_T:
            best, best_T = f, T
    return best


def load_ode(model, opt, kappa, C, alpha, beta2=0.0, etaset="base", safety=I.SAFETY):
    f = _ode_file(model, opt, float(kappa), int(C), float(alpha),
                  float(beta2 if opt == "adam" else 0.0), etaset, float(safety))
    return dict(np.load(f)) if f is not None else None


def load_sim(model, opt, kappa, C, alpha, beta2=0.0, D=1000, T=None, etaset="base"):
    r = run(model, opt, kappa, C, alpha, beta2, etaset)
    r = r._replace(horizon=T or I.sim_horizon(model, opt, C, alpha))
    path = DATA / "sim" / I.sim_path(r, D).name
    return dict(np.load(path)) if path.exists() else None


def load_frozen(kappa, C, alpha):
    path = DATA / "frozen" / f"m0_k{kappa:g}_C{C}_a{alpha:g}.npz"
    return dict(np.load(path)) if path.exists() else None


def load_frozen_bias(kappa, C, alpha):
    path = DATA / "frozen" / f"m1bias_k{kappa:g}_C{C}_a{alpha:g}.npz"
    return dict(np.load(path)) if path.exists() else None


def curves(model, opt, kappa, C, alpha, beta2=0.0, etaset="base"):
    """The trajectory used for rates: the ODE for m0, the 16-run mean for m1.
    Returns ``dict(time, loss (T, C), priors, source, [runs (n, T, C)])`` or None."""
    if model == "m0":
        d = load_ode(model, opt, kappa, C, alpha, beta2, etaset)
        if d is None:
            return None
        return {"time": d["time"], "loss": d["loss"], "priors": d["priors"], "source": "ode",
                "O": d["O"], "B": d["B"]}
    d = load_sim(model, opt, kappa, C, alpha, beta2, etaset=etaset)
    if d is None:
        return None
    out = {"time": d["time"], "loss": d["loss"].mean(0), "priors": d["priors"],
           "source": "sim", "runs": d["loss"]}
    if "beta" in d:
        out["beta"] = d["beta"]
    return out


# --------------------------------------------------------------------------- #
# Measures
# --------------------------------------------------------------------------- #
def level_times(time, loss, level=0.5):
    """``t_c``: first time ``ell_c <= level * ell_c(0)`` (NaN if never)."""
    return np.array([first_crossing(time, loss[:, c], level * loss[0, c])
                     for c in range(loss.shape[1])])


def local_exponent(time, values, smooth=3):
    """``-d log v / d log t`` (copy of ``zipf_scaling.local_exponent``)."""
    t, v = np.asarray(time[1:]), np.asarray(values[1:])
    slope = -np.gradient(np.log(v), np.log(t))
    if smooth > 1:
        kernel = np.ones(smooth) / smooth
        slope = np.convolve(np.pad(slope, smooth // 2, mode="edge"), kernel, mode="valid")
    return np.concatenate([[np.nan], slope])


def late_slopes(time, loss, decades=1.0):
    """Per-class OLS slope of ``-log ell_c`` on ``log t`` over ``[T 10^-decades, T]``."""
    m = time >= time[-1] * 10 ** -decades
    x = np.log(time[m])
    return np.array([-ols(x, np.log(loss[m, c]))[0] for c in range(loss.shape[1])])


def slope_at(time, loss, tc):
    """Local ``-d log ell_c / d log t`` evaluated at ``t_c`` (NaN if not reached)."""
    out = np.full(loss.shape[1], np.nan)
    for c in range(loss.shape[1]):
        if np.isfinite(tc[c]) and tc[c] > time[1]:
            s = local_exponent(time, loss[:, c])
            out[c] = np.interp(np.log(tc[c]), np.log(time[1:]), s[1:])
    return out


def time_lag(t_sim, R_sim, t_ode, R_ode, t_from=1.0, n=12):
    """``t_sim(R)/t_ode(R)`` at ``n`` risk levels log-spaced from ``R_ode(t_from)`` down
    to ``1.5 R_ode(T_sim)`` (past the warm-up; meaningful also on steep curves).
    Returns (levels, ratios)."""
    hi = float(np.interp(np.log(t_from), np.log(t_ode[1:]), R_ode[1:]))
    lo = 1.5 * float(np.interp(np.log(t_sim[-1]), np.log(t_ode[1:]), R_ode[1:]))
    if not hi > lo:
        return np.array([]), np.array([])
    levels = np.geomspace(hi, lo, n)
    ratio = np.array([first_crossing(t_sim, R_sim, l) / first_crossing(t_ode, R_ode, l)
                      for l in levels])
    return levels, ratio


def stall_diagnostics(time, loss):
    learned = np.isfinite(level_times(time, loss, 0.5))
    above = loss[-1] > loss[0]
    return {"learned": int(learned.sum()), "above_initial": int(above.sum()),
            "last_learned": int(np.max(np.flatnonzero(learned)) + 1) if learned.any() else 0}


def gap_stationarity(time, O):
    """Max change of the logit gaps ``O_cc - O_kc`` over ``[T/sqrt(10), T]`` relative to
    their size at ``T`` (E4's criterion; ~0 at a fixed point)."""
    k0 = int(np.argmin(np.abs(time - time[-1] / np.sqrt(10))))
    G = lambda o: np.diagonal(o)[None, :] - o
    a, b = G(O[k0]), G(O[-1])
    return float(np.abs(b - a).max() / np.abs(b).max())


def fit_set(pi, tc, horizon, factor=3.0, steep=-1.5):
    """Classes for kink fits: finite ``t_c`` well inside the horizon (``t_c <= T/10``),
    ``pi >= factor * pi_last_learned`` when any class is unlearned, and no local
    ``t_c`` slope steeper than ``steep`` (the stall zone)."""
    ok = np.isfinite(tc) & (tc <= horizon / 10)
    if not np.all(np.isfinite(tc)):
        learned = np.flatnonzero(np.isfinite(tc))
        if len(learned):
            ok &= pi >= factor * pi[learned.max()]
    if ok.sum() >= 3:
        ls = local_slopes(pi, np.where(ok, tc, np.nan))
        ok &= ~(np.isfinite(ls) & (ls < steep))
    return ok


# --------------------------------------------------------------------------- #
# Kink fits
# --------------------------------------------------------------------------- #
def ratio_break(pi, ratio, xb_grid=None):
    """Broken line in ``log(t_A/t_S)`` vs ``log pi``: slope 1/2 above ``pi_b``, 0 below
    (fixed), free level and break (profiled). Returns dict(pi_b, level, rms, n, side counts)."""
    x, y = np.log(pi), np.log(ratio)
    if xb_grid is None:
        xb_grid = np.linspace(x.min() - 2, x.max() + 2, 801)
    fit = bpl_sharp(x, y, xb_grid, slopes=(0.5, 0.0))
    xb = fit["xb"]
    return {"pi_b": float(np.exp(xb)), "level": float(np.exp(fit["y0"])),
            "rms": float(fit["rms"]), "n": int(len(x)),
            "n_above": int((x > xb).sum()), "n_below": int((x <= xb).sum()),
            "interior": bool(x.min() < xb < x.max())}


def tc_breaks(pi, tc):
    """t_c vs pi: free broken line and the smooth fixed-slope forms (p = 1, 2)."""
    x, y = np.log(pi), np.log(tc)
    xb_grid = np.linspace(x.min() - 1, x.max() + 1, 601)
    out = {}
    if len(x) >= 5:
        f = bpl_sharp(x, y, xb_grid)
        out["free"] = {"pi_b": float(np.exp(f["xb"])), "a_above": float(f["a1"]),
                       "a_below": float(f["a2"]), "rms": float(f["rms"]),
                       "interior": bool(x.min() < f["xb"] < x.max())}
    for p in (1, 2):
        f = bpl_smooth(x, y, xb_grid, p)
        out[f"smooth_p{p}"] = {"pi_b": float(np.exp(f["xb"])), "rms": float(f["rms"])}
    return out


def sbpl(p, x):
    """Smoothly broken power law in ``x = ln pi`` (``e1_rates/analysis.py``): slope ``s1``
    below, ``s2`` above ``ln pi_b``; the asymptotes intersect at ``pi_b``."""
    lnA, s1, s2, xb, lnD = p
    D = np.exp(lnD)
    z = (x - xb) / D
    return lnA + s1 * (x - xb) + (s2 - s1) * D * (np.logaddexp(0.0, z) - np.log(2.0))


def fit_sbpl(pi, rate, fixed=None):
    """Least squares in log space (copy of ``e1_rates.analysis.fit_sbpl`` with ``x`` given).
    ``fixed=(s1, s2)`` fixes the slopes."""
    X, Y = np.log(pi), np.log(rate)
    xb0, best = np.median(X), None
    for xb_start in (xb0, xb0 - 1.5, xb0 + 1.5):
        for lnD0 in (np.log(0.3), np.log(1.0)):
            if fixed is None:
                p0 = [np.interp(xb_start, np.sort(X), Y[np.argsort(X)]), 1.0, 0.5, xb_start,
                      lnD0]
                lb = [-50, -0.5, -0.5, X.min() - 5, np.log(0.05)]
                ub = [50, 2.5, 2.5, X.max() + 5, np.log(5.0)]
                fun = lambda p: sbpl(p, X) - Y
            else:
                s1, s2 = fixed
                p0 = [np.interp(xb_start, np.sort(X), Y[np.argsort(X)]), xb_start, lnD0]
                lb = [-50, X.min() - 5, np.log(0.05)]
                ub = [50, X.max() + 5, np.log(5.0)]
                fun = lambda q: sbpl([q[0], s1, s2, q[1], q[2]], X) - Y
            p0 = np.clip(p0, np.array(lb) + 1e-9, np.array(ub) - 1e-9)
            sol = least_squares(fun, p0, bounds=(lb, ub))
            if best is None or sol.cost < best.cost:
                best = sol
    p = best.x if fixed is None else np.array([best.x[0], fixed[0], fixed[1], best.x[1],
                                               best.x[2]])
    return {"s1": float(p[1]), "s2": float(p[2]), "pi_b": float(np.exp(p[3])),
            "delta": float(np.exp(p[4])), "rms_log": float(np.sqrt(2 * best.cost / len(X))),
            "p": p}


def jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return jsonable(obj.tolist())
    if isinstance(obj, (np.floating, float)):
        return None if not np.isfinite(obj) else float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    return obj
