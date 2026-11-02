"""E3 (batch size) analysis: samples needed to reach per-class, frontier and risk targets
against the batch size b, at fixed eta and with eta sqrt(b) (Adam, PSGD) or eta b (SGD).

All runs go to the same number of samples b t = 3000 (units of D), C = 64, alpha = 2,
beta2 = 0.99. numpy only; run from this directory."""
import json
import os

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
ODE = REPO + "/figs/experiments/zipf/ode"
OUT = REPO + "/figs/experiments/zipf/analysis/e3_batch"
os.makedirs(OUT, exist_ok=True)
C, SAMPLES, SAFETY = 64, 3000.0, 1.0
ETA = {"adam": 0.3, "psgd": 0.3, "sgd": 3.0}
BATCHES = (1, 4, 16, 64)
PI = np.arange(1, C + 1, dtype=float) ** -2.0
PI /= PI.sum()


def run_path(mode, eps, b, eta):
    opt = {"adam": f"b20.99_b{b}_e{eps:g}", "psgd": f"b{b}_e{eps:g}", "sgd": f"b{b}"}[mode]
    T = float(np.round(SAMPLES / b, 6))
    return f"{ODE}/{mode}_C{C}_a2_{opt}_eta{float(np.round(eta, 6)):g}_T{T:g}_s{SAFETY:g}.npz"


def load(mode, eps, b, rule):
    base = ETA[mode]
    eta = base if rule == "fixed" or b == 1 else (base * np.sqrt(b) if mode != "sgd" else base * b)
    p = run_path(mode, eps, b, eta)
    return (dict(np.load(p)), eta) if os.path.exists(p) else (None, eta)


def first(t, v, level):
    k = np.flatnonzero(v <= level)
    if not len(k):
        return np.nan
    k = k[0]
    if k == 0:
        return t[0]
    w = (v[k - 1] - level) / max(v[k - 1] - v[k], 1e-300)
    return float(np.exp(np.log(max(t[k - 1], 1e-12)) + np.clip(w, 0, 1) * (np.log(t[k]) - np.log(max(t[k - 1], 1e-12)))))


results = {}
targets_k = (8, 16, 32, 48, 64)
targets_R = (0.2, 0.1, 0.05, 0.03, 0.02, 0.015, 0.01)
for eps in (1e-8, 1e-4):
    for mode in ("adam", "psgd", "sgd"):
        if mode == "sgd" and eps != 1e-8:
            continue
        rules = ("fixed", "linear") if mode == "sgd" else ("fixed", "sqrt")
        for rule in rules:
            for b in BATCHES:
                r, eta = load(mode, eps, b, rule)
                if r is None:
                    continue
                t, L = r["time"], r["loss"]
                tc = np.array([first(t, L[:, c], L[0, c] / 2) for c in range(C)])
                S = b * tc                                            # samples / D to learn class c
                frontier = {k: (np.sort(np.where(np.isfinite(S), S, np.inf))[k - 1]) for k in targets_k}
                R = L @ PI
                SR = {R0: b * first(t, R, R0) for R0 in targets_R}
                results[(mode, eps, rule, b)] = dict(eta=eta, S=S, frontier=frontier, SR=SR, R_end=R[-1],
                                                     learned=int(np.isfinite(tc).sum()))

print("Samples (b t, units of D) until k classes are learned; inf = not within b t = 3000\n")
for (mode, eps, rule) in sorted({k[:3] for k in results}):
    print(f"{mode} eps={eps:g} {rule}:")
    for b in BATCHES:
        r = results.get((mode, eps, rule, b))
        if r is None:
            continue
        print(f"   b={b:<3d} eta={r['eta']:<6.3g} learned {r['learned']:2d}/64, R_end {r['R_end']:.4f} | k="
              + " ".join(f"{k}:{r['frontier'][k]:8.1f}" for k in targets_k)
              + " | R<=" + " ".join(f"{R0:g}:{r['SR'][R0]:7.1f}" for R0 in targets_R))

print("\nBest batch size per target (samples-minimizing b among 1, 4, 16, 64) and the samples ratio max/min:")
summary = {}
for (mode, eps, rule) in sorted({k[:3] for k in results}):
    rows = []
    for k in targets_k:
        vals = {b: results[(mode, eps, rule, b)]["frontier"][k] for b in BATCHES if (mode, eps, rule, b) in results}
        fin = {b: v for b, v in vals.items() if np.isfinite(v)}
        if fin:
            bb = min(fin, key=fin.get)
            rows.append(f"k={k}: b*={bb} ({fin[bb]:.1f}; " + ", ".join(f"b{b}:{v:.0f}" for b, v in vals.items()) + ")")
            summary[f"{mode}_eps{eps:g}_{rule}_k{k}"] = dict(best_b=int(bb), samples={str(b): float(v) for b, v in vals.items()})
    print(f"{mode} eps={eps:g} {rule}: " + "; ".join(rows))

print("\nPer-class samples to learn, Adam fixed eta, classes 4, 16, 32, 64:")
for eps in (1e-8, 1e-4):
    for rule in ("fixed", "sqrt"):
        line = []
        for b in BATCHES:
            r = results.get(("adam", eps, rule, b))
            if r is not None:
                line.append(f"b{b}: " + " ".join(f"{r['S'][c]:7.1f}" for c in (3, 15, 31, 63)))
        print(f"  eps={eps:g} {rule}: " + " | ".join(line))

json.dump(summary, open(OUT + "/e3_numbers.json", "w"), indent=1)
print("\nwrote", OUT + "/e3_numbers.json")
