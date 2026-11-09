"""Series J Stage 4: loading S3a groups and per-lane phase markers (numpy only).

A *group* is one ``figs/experiments/knee/runs/J_*`` directory: one optimizer, ``C``, ``alpha``,
``kappa`` (and ``eps`` for Adam), ``D``, 32 runs, and for Adam four ``beta2`` lanes. Records sit on
``t = 0`` plus the grid ``0.01 * 10^(k/16)`` (finer at the start for large ``D``).

Curves are seed means over the 32 runs (``loss_cv``: the control-variate loss, equal to the
plain MC loss where the variate is not used). At ``alpha = 0`` all classes are exchangeable, so
markers are taken on the class mean; the class spread is reported as a check.

Phase markers of a curve ``ell(t)`` (``ell0 = ell(0)``):

* ``t_plateau``: first ``ell <= 0.9 ell0``; ``t_c``: first ``ell <= ell0/2``;
* ``knee``: the maximum of the local exponent ``e(t) = -d ln ell/d ln t`` after ``t_c``
  (3-record smoothing); ``t_knee``, ``ell_knee/ell0``, ``m_knee`` (seed-mean class-mean margin),
  ``frac_eps`` there;
* ``cliff``: on ``[t_c, t_knee]`` the OLS slope of ``ln ell`` against linear ``t`` and the
  ``R^2`` of that fit and of ``ln ell`` against ``ln t`` (cliff linearity);
* ``tail``: OLS exponent of ``ell`` over the last decade ``[T/10, T]`` and over the previous one.
"""
from __future__ import annotations

import glob
import json
from functools import lru_cache
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
DATA = REPO / "figs" / "experiments" / "knee"
RUNS = DATA / "runs"
OUTDIR = DATA / "analysis"
FIELD = OUTDIR / "field"
R = 5.0
ETA = {"adam": 0.3, "sgd": 3.0}
BETA2S = (0.9, 0.95, 0.99, 0.999)
KAPPAS = (2.0, 1.0, 0.5, 0.25, 0.1)


# --------------------------------------------------------------------------- #
# Groups
# --------------------------------------------------------------------------- #
def parse(name):
    """``J_{opt}_C{C}_a{alpha}_k{kappa}[_e{eps}]_D{D}_R{runs}_s{seed}_{dtype}_u{U}[_b{beta2s}]``."""
    p = name.split("_")
    out = {"name": name, "opt": p[1], "C": int(p[2][1:]), "alpha": float(p[3][1:]),
           "kappa": float(p[4][1:])}
    rest = p[5:]
    if out["opt"] == "adam":
        out["eps"], rest = float(rest[0][1:]), rest[1:]
    else:
        out["eps"] = None
    out["D"] = int(rest[0][1:])
    out["lane_tag"] = rest[-1] if rest[-1].startswith("b") else ""
    return out


def groups(alpha=0.0):
    return [parse(p.name) for p in sorted(RUNS.iterdir()) if p.is_dir()
            and parse(p.name)["alpha"] == alpha]


def find(opt, C, kappa, eps=None, alpha=0.0, D=None, lane_tag=""):
    """The main group for a configuration (default ``D`` rule: the largest-D duplicate is the
    finite-D check only if requested explicitly)."""
    cands = [g for g in groups(alpha) if g["opt"] == opt and g["C"] == C
             and g["kappa"] == kappa and (opt == "sgd" or g["eps"] == eps)
             and g["lane_tag"] == lane_tag]
    if D is not None:
        cands = [g for g in cands if g["D"] == D]
    else:
        d_rule = max(1000, int(np.ceil(20 * R ** 2 / kappa)))
        main = [g for g in cands if g["D"] == d_rule]
        cands = main or cands
    return cands[0] if cands else None


@lru_cache(maxsize=None)
def load(name):
    """Concatenated chunks: ``time (T,)``, ``loss, margin, frac_eps (T, L, runs, C)``,
    ``B_mean, O_mean (T, L, C, C)``, ``priors``, ``beta2s``."""
    fs = sorted(glob.glob(str(RUNS / name / "chunk_*.npz")))
    keys = ("time", "loss_cv", "loss", "margin_mean", "frac_eps", "B_mean", "O_mean")
    acc = {k: [] for k in keys}
    for f in fs:
        d = np.load(f)
        for k in keys:
            acc[k].append(d[k])
        priors, beta2s = d["priors"], d["beta2s"]
    out = {k: np.concatenate(v) for k, v in acc.items()}
    # early records can repeat a step (rounded log grid at small D*t): keep the first
    keep = np.concatenate([[True], np.diff(out["time"]) > 0])
    out = {k: v[keep] for k, v in out.items()}
    out["loss_mc"] = out.pop("loss")
    out["loss"] = out.pop("loss_cv")
    out["margin"] = out.pop("margin_mean")
    out["priors"], out["beta2s"] = priors, beta2s
    return out


def status():
    """Last certificate record per group from ``status.jsonl``."""
    last = {}
    for line in (DATA / "status.jsonl").read_text().splitlines():
        r = json.loads(line)
        last[r["group"]] = r
    return last


def optimum(C, kappa):
    o = json.loads((OUTDIR / "optimum.json").read_text())[f"C{C}_a0_k{kappa:g}"]
    return {"ell": float(np.mean(o["ell"])), "m": float(np.mean(o["m"])),
            "ell_rel": float(np.mean(o["ell_rel"]))}


def frozen_zero(C, beta2, kappa):
    """K2 drift zeros of ``phi_alpha0.json`` (computed at kappa = 1) scaled by ``r'^2/25 = 1/kappa``;
    returns (low, high) over the two noise levels of the synthetic family, or None."""
    fp = json.loads((OUTDIR / "phi_alpha0.json").read_text())[f"C{C}_b{beta2:g}"]["fixed_point"]
    ms = [v.get("crossing_m") for v in fp.values() if v.get("crossing_m") is not None]
    if not ms:
        return None
    return min(ms) / kappa, max(ms) / kappa


# --------------------------------------------------------------------------- #
# Curves and markers
# --------------------------------------------------------------------------- #
def first_crossing(t, y, level):
    """First ``t`` with ``y <= level``, log-linear interpolation (``t[0] = 0`` handled linearly)."""
    below = np.flatnonzero(y <= level)
    if not len(below):
        return np.nan
    k = below[0]
    if k == 0:
        return float(t[0])
    t0, t1 = t[k - 1], t[k]
    w = np.clip((y[k - 1] - level) / max(y[k - 1] - y[k], 1e-300), 0, 1)
    if t0 > 0:
        return float(np.exp(np.log(t0) + w * (np.log(t1) - np.log(t0))))
    return float(t0 + w * (t1 - t0))


def local_exponent(t, y, smooth=3):
    """``-d ln y / d ln t`` on records with ``t > 0`` (centred differences, box smoothing)."""
    k = t > 0
    out = np.full(len(t), np.nan)
    s = -np.gradient(np.log(np.maximum(y[k], 1e-300)), np.log(t[k]))
    if smooth > 1:
        s = np.convolve(np.pad(s, smooth // 2, mode="edge"), np.ones(smooth) / smooth, "valid")
    out[k] = s
    return out


def ols(x, y):
    X = np.column_stack([np.ones_like(x), x])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    res = y - X @ coef
    ss = np.sum((y - y.mean()) ** 2)
    return float(coef[1]), float(1 - res @ res / ss) if ss > 0 else np.nan


def markers(t, ell, margin=None, frac_eps=None):
    """Phase markers of one curve (see module docstring)."""
    ell0 = ell[0]
    out = {"ell0": float(ell0), "T": float(t[-1]), "ell_T_rel": float(ell[-1] / ell0),
           "t_plateau": first_crossing(t, ell, 0.9 * ell0), "t_c": first_crossing(t, ell, 0.5 * ell0)}
    e = local_exponent(t, ell)
    out["exp_T"] = float(e[-1])
    kk = None
    if np.isfinite(out["t_c"]):
        after = np.flatnonzero((t >= out["t_c"]) & np.isfinite(e))
        # knee = the first local maximum of e after t_c that is followed, within half a decade
        # (8 records), by a drop to <= 0.8 of it: the end of the cliff. Later steepenings (an
        # eps-floor escape from a K2 balance) are reported by the tail exponents instead.
        for i in after:
            nxt = e[i + 1:i + 9]
            if i > 0 and len(nxt) and e[i] >= e[i - 1] and e[i] >= nxt[0] and \
                    np.nanmin(nxt) <= 0.8 * e[i]:
                kk = i
                break
        if kk is None and len(after):
            kk = after[np.argmax(e[after])]
            out["knee_fallback"] = True
    if kk is not None:
        out.update({"t_knee": float(t[kk]), "exp_knee": float(e[kk]),
                    "ell_knee_rel": float(ell[kk] / ell0), "k_knee": int(kk)})
        if margin is not None:
            out["m_knee"] = float(margin[kk])
        if frac_eps is not None:
            out["frac_eps_knee"] = float(frac_eps[kk])
        w = (t >= out["t_c"]) & (t <= t[kk]) & (t > 0)
        if w.sum() >= 4:
            out["cliff_slope_t"], out["R2_lin_t"] = ols(t[w], np.log(ell[w]))
            _, out["R2_log_t"] = ols(np.log(t[w]), np.log(ell[w]))
            out["cliff_span"] = float(t[w][-1] / t[w][0])
    T = t[-1]
    for name, lo, hi in (("tail_last", T / 10, T), ("tail_prev", T / 100, T / 10)):
        w = (t >= lo) & (t <= hi) & (t > 0)
        if w.sum() >= 4:
            out[name] = -ols(np.log(t[w]), np.log(ell[w]))[0]
    if margin is not None:
        out["m_T"] = float(margin[-1])
    if frac_eps is not None:
        out["frac_eps_T"] = float(frac_eps[-1])
    return out


def lane_curves(d, lane):
    """Seed-mean class-mean loss, margin and frac_eps of one lane, plus the per-class seed means."""
    ell_c = d["loss"][:, lane].mean(1)                      # (T, C)
    return {"t": d["time"], "ell": ell_c.mean(-1), "ell_c": ell_c,
            "margin": d["margin"][:, lane].mean((1, 2)),
            "frac_eps": d["frac_eps"][:, lane].mean((1, 2))}


def lane_markers(d, lane, boot=0, seed=0):
    """Markers of a lane, with ``boot`` bootstrap replicates over runs (percentile 5-95 CI)."""
    c = lane_curves(d, lane)
    out = markers(c["t"], c["ell"], c["margin"], c["frac_eps"])
    tc_cls = np.array([first_crossing(c["t"], c["ell_c"][:, k], 0.5 * c["ell_c"][0, k])
                       for k in range(c["ell_c"].shape[1])])
    out["t_c_class_spread"] = float(np.nanstd(np.log(tc_cls))) if np.isfinite(tc_cls).any() else np.nan
    if boot:
        rng = np.random.default_rng(seed)
        nr = d["loss"].shape[2]
        reps = {k: [] for k in ("t_c", "t_knee", "ell_knee_rel", "m_knee", "ell_T_rel",
                                "tail_last", "cliff_slope_t")}
        for _ in range(boot):
            idx = rng.integers(0, nr, nr)
            ell = d["loss"][:, lane][:, idx].mean((1, 2))
            mg = d["margin"][:, lane][:, idx].mean((1, 2))
            m = markers(d["time"], ell, mg)
            for k in reps:
                reps[k].append(m.get(k, np.nan))
        for k, v in reps.items():
            v = np.asarray(v, float)
            if np.isfinite(v).sum() > boot // 2:
                out[f"{k}_ci"] = [float(np.nanpercentile(v, 5)), float(np.nanpercentile(v, 95))]
    return out


def jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return jsonable(obj.tolist())
    if isinstance(obj, (float, np.floating)):
        return float(obj) if np.isfinite(obj) else None
    if isinstance(obj, np.integer):
        return int(obj)
    return obj
