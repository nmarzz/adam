"""Independent recomputation of the headline numbers in ``numbers.json``.

Separate code paths from ``lib``/``analyze``: raw npz files are read directly, t_c is
found with a vectorised crossing search, the Adam/SGD ratio break is fitted with
``scipy.optimize.least_squares`` on a softplus-free hinge (continuous, slopes fixed)
from several starts, and risks and frozen kinks are recomputed. Prints the largest
discrepancies; anything above the tolerances is flagged.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

HERE = Path(__file__).resolve().parent
DATA = HERE.parents[2] / "figs" / "experiments" / "imbalance"
NUM = json.loads((DATA / "analysis" / "numbers.json").read_text())
FLAGS = []


def tc_vec(t, L, level=0.5):
    """First crossing of L[:, c] <= level * L[0, c], log-linear interpolation."""
    target = level * L[0]
    below = L <= target[None, :]
    out = np.full(L.shape[1], np.nan)
    for c in range(L.shape[1]):
        idx = np.argmax(below[:, c])
        if not below[idx, c]:
            continue
        if idx == 0:
            out[c] = t[0]
            continue
        lt0, lt1 = np.log(t[idx - 1]) if t[idx - 1] > 0 else -np.inf, np.log(t[idx])
        y0, y1 = L[idx - 1, c], L[idx, c]
        w = np.clip((y0 - target[c]) / (y0 - y1), 0, 1)
        out[c] = np.exp(lt0 + w * (lt1 - lt0)) if np.isfinite(lt0) else t[idx - 1] + w * (
            t[idx] - t[idx - 1])
    return out


def hinge_fit(x, y):
    """y = y0 + 0.5 * max(x - xb, 0): least squares over (y0, xb), many starts."""
    best = None
    for xb0 in np.linspace(x.min() - 1, x.max() + 1, 25):
        fun = lambda p: p[0] + 0.5 * np.maximum(x - p[1], 0) - y
        sol = least_squares(fun, [np.median(y), xb0], method="trf",
                            bounds=([-50, x.min() - 2], [50, x.max() + 2]))
        if best is None or sol.cost < best.cost:
            best = sol
    return float(np.exp(best.x[1]))


def compare(name, a, b, tol):
    a, b = np.asarray(a, float), np.asarray(b, float)
    both = np.isfinite(a) & np.isfinite(b)
    if (np.isfinite(a) != np.isfinite(b)).any():
        FLAGS.append(f"{name}: finiteness differs")
    d = float(np.max(np.abs(np.log(a[both] / b[both])))) if both.any() else 0.0
    if d > tol:
        FLAGS.append(f"{name}: max |log ratio| {d:.2e} > {tol:g}")
    return d


def main():
    worst = {"tc": 0.0, "RT": 0.0, "ratio_pi_b": 0.0}
    for k, rec in NUM["runs"].items():
        model, kappa, C, alpha, opt = k.split("_")
        if rec["source"] == "ode":
            pattern = (f"m0_{'adam' if opt.startswith('adam') else 'sgd'}_k{kappa}_C{C}_a{alpha}_"
                       + (f"b10.9_b2{opt[4:]}_b1_e1e-08_eta0.3" if opt.startswith("adam")
                          else "b1_eta3") + "_T*_s1.npz")
            files = sorted((DATA / "ode").glob(pattern))
            files = [f for f in files if abs(float(np.load(f)["time"][-1]) - rec["T"]) < 1e-6]
            d = np.load(files[0])
            t, L, pi = d["time"], d["loss"], d["priors"]
        else:
            pattern = (f"{model}_{'adam' if opt.startswith('adam') else 'sgd'}_k{kappa}_C{C}_a{alpha}_"
                       + (f"b10.9_b2{opt[4:]}_b1_e1e-08_eta0.3" if opt.startswith("adam")
                          else "b1_eta3") + "_T*_D1000.npz")
            files = [f for f in sorted((DATA / "sim").glob(pattern))
                     if abs(float(np.load(f)["time"][-1]) - rec["T"]) < 1e-6]
            d = np.load(files[0])
            t, L, pi = d["time"], d["loss"].mean(0), d["priors"]
        worst["tc"] = max(worst["tc"], compare(f"tc {k}", tc_vec(t, L), rec["tc"]["0.5"], 1e-9))
        worst["RT"] = max(worst["RT"], compare(f"R(T) {k}", L[-1] @ pi, rec["RT"], 1e-12))
    for k, rec in NUM["kink"].items():
        if "ratio" not in rec:
            continue
        model, kappa, C, alpha, b2 = k.split("_")
        a = NUM["runs"][f"{model}_{kappa}_{C}_{alpha}_adam{b2}"]
        s = NUM["runs"][f"{model}_{kappa}_{C}_{alpha}_sgd"]
        idx = np.asarray(rec["classes"]) - 1
        pi = np.asarray(a["pi"])[idx]
        r = np.asarray(a["tc"]["0.5"])[idx] / np.asarray(s["tc"]["0.5"])[idx]
        pb = hinge_fit(np.log(pi), np.log(r))
        # profile-grid resolution of the analyze fit is (range + 4) / 800 in log pi
        tol = (np.log(pi).max() - np.log(pi).min() + 4) / 800 * 1.5
        worst["ratio_pi_b"] = max(worst["ratio_pi_b"],
                                  compare(f"ratio pi_b {k}", pb, rec["ratio"]["pi_b"], tol))
    print("largest |log| discrepancies:", {k: f"{v:.2e}" for k, v in worst.items()})
    print("FLAGS:" if FLAGS else "no flags", *FLAGS, sep="\n  ")


if __name__ == "__main__":
    main()
