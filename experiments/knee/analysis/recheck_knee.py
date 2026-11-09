"""Independent recheck of the S3a numbers in ``s3a_numbers.json`` from the raw chunk files.

Separate code path from ``lib_knee``: chunks are read one by one with their own de-duplication,
the half-loss time is found by ``np.interp`` on the reversed monotone envelope, the local
exponent by explicit forward/backward differences with a uniform 3-record moving average, and
the knee by the same written-out rule (first local maximum after ``t_c`` followed within 8 records
by a drop to <= 0.8). Prints the largest discrepancies; flags anything above tolerance.
Run: ``real/bin/python experiments/knee/analysis/recheck_knee.py``.
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3] / "figs" / "experiments" / "knee"
NUM = json.loads((ROOT / "analysis" / "s3a_numbers.json").read_text())
FLAGS = []


def read(name):
    t, loss, marg = [], [], []
    for f in sorted(glob.glob(str(ROOT / "runs" / name / "chunk_*.npz"))):
        z = np.load(f)
        for i in range(len(z["time"])):
            if t and z["time"][i] <= t[-1]:
                continue
            t.append(float(z["time"][i]))
            loss.append(z["loss_cv"][i].mean(1).mean(-1))     # (lanes,)
            marg.append(z["margin_mean"][i].mean((1, 2)))
        beta2s = z["beta2s"]
    return np.array(t), np.array(loss), np.array(marg), beta2s


def half_time(t, y):
    level = 0.5 * y[0]
    idx = np.nonzero(y <= level)[0]
    if idx.size == 0:
        return np.nan
    i = idx[0]
    if i == 0:
        return t[0]
    a, b = np.log(t[i - 1]) if t[i - 1] > 0 else None, np.log(t[i])
    frac = (y[i - 1] - level) / (y[i - 1] - y[i])
    frac = min(max(frac, 0.0), 1.0)
    return float(np.exp(a + frac * (b - a))) if a is not None else float(t[i - 1] + frac * (t[i] - t[i - 1]))


def exponent(t, y):
    lt, ly = np.log(t[1:]), np.log(y[1:])
    e = np.empty_like(lt)
    e[1:-1] = -((ly[2:] - ly[1:-1]) / (lt[2:] - lt[1:-1]) * (lt[1:-1] - lt[:-2])
                + (ly[1:-1] - ly[:-2]) / (lt[1:-1] - lt[:-2]) * (lt[2:] - lt[1:-1])) \
        / (lt[2:] - lt[:-2])
    e[0] = -(ly[1] - ly[0]) / (lt[1] - lt[0])
    e[-1] = -(ly[-1] - ly[-2]) / (lt[-1] - lt[-2])
    padded = np.concatenate([[e[0]], e, [e[-1]]])
    sm = (padded[:-2] + padded[1:-1] + padded[2:]) / 3
    return np.concatenate([[np.nan], sm])


def knee(t, y, tc):
    if not np.isfinite(tc):
        return np.nan
    e = exponent(t, y)
    cand = [i for i in range(1, len(t)) if t[i] >= tc and np.isfinite(e[i])]
    for i in cand:
        nxt = e[i + 1:i + 9]
        if len(nxt) and e[i] >= e[i - 1] and e[i] >= nxt[0] and np.nanmin(nxt) <= 0.8 * e[i]:
            return t[i]
    return t[cand[int(np.argmax(e[cand]))]] if cand else np.nan


def cmp(label, a, b, tol):
    if a is None or b is None or not (np.isfinite(a) and np.isfinite(b)):
        if (a is None or not np.isfinite(a)) != (b is None or not np.isfinite(b)):
            FLAGS.append(f"{label}: finite in one only ({a}, {b})")
        return 0.0
    d = abs(np.log(a / b)) if a > 0 and b > 0 else abs(a - b)
    if d > tol:
        FLAGS.append(f"{label}: {a:.6g} vs {b:.6g}")
    return d


def main():
    worst = {"t_c": 0.0, "t_knee": 0.0, "ell_T": 0.0, "m_T": 0.0}
    groups = sorted(p.name for p in (ROOT / "runs").iterdir() if p.is_dir() and "_a0_" in p.name)
    for name in groups:
        p = name.split("_")
        opt, C, kappa = p[1], int(p[2][1:]), float(p[4][1:])
        eps = p[5][1:] if opt == "adam" else None
        D = [x for x in p if x.startswith("D")][0]
        tag = "_" + p[-1] if p[-1].startswith("b") else ""
        t, loss, marg, beta2s = read(name)
        for lane in range(loss.shape[1]):
            ln = "sgd" if opt == "sgd" else f"{beta2s[lane]:g}"
            key = f"C{C}_k{kappa:g}{'_e' + f'{float(eps):g}' if eps else ''}_{D}{tag}_{ln}"
            rec = NUM["markers"].get(key)
            if rec is None:
                FLAGS.append(f"missing {key}")
                continue
            y = loss[:, lane]
            tc = half_time(t, y)
            worst["t_c"] = max(worst["t_c"], cmp(f"t_c {key}", tc, rec.get("t_c"), 1e-9))
            worst["t_knee"] = max(worst["t_knee"], cmp(f"t_knee {key}", knee(t, y, tc),
                                                        rec.get("t_knee"), 1e-9))
            worst["ell_T"] = max(worst["ell_T"], cmp(f"ell_T {key}", y[-1] / y[0],
                                                      rec.get("ell_T_rel"), 1e-9))
            worst["m_T"] = max(worst["m_T"], cmp(f"m_T {key}", marg[-1, lane], rec.get("m_T"), 1e-9))
    print("largest discrepancies (|log ratio| or abs):", {k: f"{v:.2e}" for k, v in worst.items()})
    print("FLAGS:" if FLAGS else "no flags", *FLAGS[:30], sep="\n  ")


if __name__ == "__main__":
    main()
