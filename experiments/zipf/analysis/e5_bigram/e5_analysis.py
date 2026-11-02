"""E5 (linear bigram model) analysis: per-token learning times against the token frequency,
band slopes, a two-segment fit of the break against the predicted beta2 kink
kappa_beta (1 - beta2)/b, and the local exponent of the risk. numpy only; run from this directory."""
import glob
import os
import re

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
OUT = REPO + "/figs/experiments/zipf"
BETA1 = 0.9


def kappa(beta2):
    return (np.pi / 2) * ((1 - BETA1 / np.sqrt(beta2)) / (1 - BETA1)) ** 2


def first(steps, v, level):
    k = np.flatnonzero(v <= level)
    if not len(k):
        return np.nan
    k = k[0]
    if k == 0:
        return float(steps[0])
    w = (v[k - 1] - level) / max(v[k - 1] - v[k], 1e-300)
    lo = np.log(max(steps[k - 1], 0.5))
    return float(np.exp(lo + np.clip(w, 0, 1) * (np.log(steps[k]) - lo)))


def two_segment(x, y):
    """Continuous broken line in (log pi, log t) with the break profiled; returns
    (break pi, slope above, slope below, rms)."""
    best = None
    for xb in np.linspace(np.quantile(x, 0.1), np.quantile(x, 0.9), 60):
        A = np.stack([np.ones_like(x), np.minimum(x - xb, 0), np.maximum(x - xb, 0)], 1)
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        rms = np.sqrt(np.mean((A @ coef - y) ** 2))
        if best is None or rms < best[3]:
            best = (np.exp(xb), coef[2], coef[1], rms)
    return best


bands = ((1, 6), (6, 40), (40, 250), (250, 1001))               # token ranks [lo, hi)
print(f"{'run':38s} {'learned':>8s} | band slopes of log t_i vs log pi_i for ranks "
      + ", ".join(f"[{lo},{hi})" for lo, hi in bands) + " | break pi, /kink, slopes above/below | risk exponent")
for f in sorted(glob.glob(OUT + "/bigram_*.npz")):
    name = os.path.basename(f)[:-4]
    loss, alpha, opt, b2, b = re.match(r"bigram_(\w+?)_V\d+_a([\d.]+)_(adam_b2([\d.]+)|sgd)_b(\d+)", name).group(1, 2, 3, 4, 5)
    b = int(b)
    d = np.load(f)
    steps, pi, L = d["steps"], d["priors"], d["loss"]                    # L: (seeds, T, V)
    t = np.array([[first(steps, Ls[:, i], Ls[0, i] / 2) for i in range(L.shape[2])] for Ls in L])
    tm = np.exp(np.nanmean(np.log(t), 0))                               # geometric mean over seeds
    learned = np.isfinite(t).all(0).sum()
    slopes = []
    for lo, hi in bands:
        rank = np.arange(1, len(pi) + 1)
        m = (rank >= lo) & (rank < hi) & np.isfinite(tm)
        slopes.append(np.polyfit(np.log(pi[m]), np.log(tm[m]), 1)[0] if m.sum() >= 4 else np.nan)
    m = np.isfinite(tm)
    pb, up, down, rms = two_segment(np.log(pi[m]), np.log(tm[m]))
    kink = kappa(float(b2)) * (1 - float(b2)) / b if opt.startswith("adam") else np.nan
    R = L.mean(0) @ pi
    ok = (R > 0) & (steps > 0)
    win = ok & (R < 0.5 * R[0]) & (R > 1e-3 * R[0])
    rexp = -np.polyfit(np.log(steps[win] * b), np.log(R[win]), 1)[0] if win.sum() > 3 else np.nan
    print(f"{name:38s} {learned:5d}/{L.shape[2]} | " + " ".join(f"{s:+6.2f}" for s in slopes)
          + f" | {pb:8.2e} {pb / kink if np.isfinite(kink) else np.nan:5.2f} {up:+5.2f}/{down:+5.2f} | {rexp:5.2f}")
