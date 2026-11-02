"""Helpers for the E1b analysis (per-class learning times) of timescales.npz.

Read-only on the repository data; numpy only (no jax)."""
import os
import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
DATA = REPO + "/figs/experiments/zipf/timescales.npz"
SCR = REPO + "/figs/experiments/zipf/analysis/e1_timescales"
os.makedirs(SCR, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", "/tmp/adam-matplotlib")

BETA1, ETA_ADAM, ETA_SGD = 0.9, 0.3, 3.0

# key -> (beta2, b, eps); PSGD does not depend on beta2 (identical arrays for 0.95 / 0.99)
ADAM = {
    "adam_b20.99_b1_e1e-08": (0.99, 1, 1e-8),
    "adam_b20.95_b1_e1e-08": (0.95, 1, 1e-8),
    "adam_b20.99_b16_e1e-08": (0.99, 16, 1e-8),
    "adam_b20.99_b1_e0.0001": (0.99, 1, 1e-4),
}
PSGD = {k.replace("adam", "psgd"): v for k, v in ADAM.items()}
SGD = {"sgd_b1": (None, 1, None), "sgd_b16": (None, 16, None)}
SHORT = {
    "adam_b20.99_b1_e1e-08": "Adam b2=.99 b=1",
    "adam_b20.95_b1_e1e-08": "Adam b2=.95 b=1",
    "adam_b20.99_b16_e1e-08": "Adam b2=.99 b=16",
    "adam_b20.99_b1_e0.0001": "Adam b2=.99 b=1 eps=1e-4",
    "psgd_b20.99_b1_e1e-08": "PSGD b=1",
    "psgd_b20.95_b1_e1e-08": "PSGD b=1 (b2=.95 copy)",
    "psgd_b20.99_b16_e1e-08": "PSGD b=16",
    "psgd_b20.99_b1_e0.0001": "PSGD b=1 eps=1e-4",
    "sgd_b1": "SGD b=1",
    "sgd_b16": "SGD b=16",
}


def load():
    return dict(np.load(DATA))


def kappa_beta(beta1, beta2):
    """Same formula as experiments/zipf_scaling.kappa_beta."""
    return (np.pi / 2) * ((1 - beta1 / np.sqrt(beta2)) / (1 - beta1)) ** 2


def kink_beta2(beta2, b, beta1=BETA1):
    return kappa_beta(beta1, beta2) * (1 - beta2) / b


def first_crossing(times, values, level):
    """Copy of experiments/zipf_scaling.first_crossing (log-linear interpolation)."""
    times, values = np.asarray(times), np.asarray(values)
    below = np.flatnonzero(values <= level)
    if not len(below):
        return np.nan
    k = below[0]
    if k == 0:
        return times[0]
    t0, t1 = times[k - 1], times[k]
    w = (values[k - 1] - level) / max(values[k - 1] - values[k], 1e-300)
    if t0 > 0:
        return float(np.exp(np.log(t0) + np.clip(w, 0, 1) * (np.log(t1) - np.log(t0))))
    return float(t0 + np.clip(w, 0, 1) * (t1 - t0))


def interp_log(t_new, t, y):
    """Copy of experiments/zipf_scaling.interp_log (first axis, in log t)."""
    t_new, t, y = np.asarray(t_new), np.asarray(t), np.asarray(y)
    x = np.log(np.maximum(t, 1e-12))
    xn = np.log(np.maximum(t_new, 1e-12))
    flat = y.reshape(len(t), -1)
    out = np.stack([np.interp(xn, x, flat[:, j]) for j in range(flat.shape[1])], axis=-1)
    return out.reshape((len(t_new),) + y.shape[1:])


def ols(x, y):
    """Slope, intercept, slope s.e., rms residual of y = a + s x."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    X = np.column_stack([np.ones_like(x), x])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    res = y - X @ coef
    n = len(x)
    s2 = res @ res / max(n - 2, 1)
    cov = s2 * np.linalg.inv(X.T @ X)
    return coef[1], coef[0], float(np.sqrt(cov[1, 1])), float(np.sqrt(np.mean(res ** 2)))


def band_slope(pi, tc, classes):
    """log-log slope of t_c vs pi_c over 1-based class list."""
    idx = np.asarray(classes) - 1
    ok = np.isfinite(tc[idx])
    idx = idx[ok]
    return ols(np.log(pi[idx]), np.log(tc[idx])) + (len(idx),)


def bpl_sharp(x, y, xb_grid, slopes=None):
    """Continuous broken line in (x, y) = (log pi, log t): slope a1 for x > xb, a2 for
    x < xb. Free slopes (slopes=None) or fixed (a1, a2). Profile over xb.
    Returns dict(xb, y0, a1, a2, rms, sse_profile)."""
    best, prof = None, []
    for xb in xb_grid:
        hi, lo = np.maximum(x - xb, 0), np.minimum(x - xb, 0)
        if slopes is None:
            X = np.column_stack([np.ones_like(x), hi, lo])
            coef, *_ = np.linalg.lstsq(X, y, rcond=None)
            pred = X @ coef
            y0, a1, a2 = coef
        else:
            a1, a2 = slopes
            y0 = np.mean(y - a1 * hi - a2 * lo)
            pred = y0 + a1 * hi + a2 * lo
        sse = float(np.sum((y - pred) ** 2))
        prof.append(sse)
        if best is None or sse < best["sse"]:
            best = dict(xb=xb, y0=y0, a1=a1, a2=a2, sse=sse)
    best["rms"] = np.sqrt(best["sse"] / len(x))
    best["profile"] = np.array(prof)
    return best


def bpl_smooth(x, y, xb_grid, p):
    """t = A pi^{-1/2} (1 + (pb/pi)^{p/2})^{1/p}: theory slopes -1/2 and -1, asymptotes
    meet at pb; p = 1 is t = A pi^{-1/2} + A' pi^{-1}, p = 2 quadrature. Profile over xb."""
    best = None
    for xb in xb_grid:
        g = -0.5 * x + np.log1p(np.exp(p / 2 * (xb - x))) / p
        y0 = np.mean(y - g)
        sse = float(np.sum((y - y0 - g) ** 2))
        if best is None or sse < best["sse"]:
            best = dict(xb=xb, logA=y0, sse=sse)
    best["rms"] = np.sqrt(best["sse"] / len(x))
    return best


def local_slopes(pi, tc, half_width=0.2, min_pts=3):
    """Sliding-window OLS slope of log t on log pi within +-half_width decades."""
    lp = np.log10(pi)
    out = np.full(len(pi), np.nan)
    for c in range(len(pi)):
        m = (np.abs(lp - lp[c]) <= half_width) & np.isfinite(tc)
        if m.sum() >= min_pts:
            out[c] = ols(np.log(pi[m]), np.log(tc[m]))[0]
    return out
