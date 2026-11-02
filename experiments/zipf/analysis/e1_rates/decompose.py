"""Split the frozen Adam rate into forcing r^2 pi_c E[psi_c(1 - p_c(q;c))] and the
restoring part -(phi^T O)_cc, using the saved psi / dpsi tables (last seed) and
Gaussian draws q ~ N(O[:, c], B) (numpy only).  Usage: decompose.py init|mid
Saves decomp_<state>.npz."""
import sys, time
import numpy as np
sys.path.insert(0, __file__.rsplit("/", 1)[0])
from common import *

state = sys.argv[1]
r = load()
pi = r["priors"]; C = len(pi); fg = r["f_grid"]
f0 = 1e-4
u_grid = np.arcsinh(fg / f0)
n = int(sys.argv[2]) if len(sys.argv) > 2 else (2 ** 11 if state == "init" else 2 ** 13)
draws = np.random.default_rng(0).standard_normal((n, C))
FULL = sys.argv[3].split(",") if len(sys.argv) > 3 else [
    "b20.99_b1_e1e-08", "b20.99_b16_e1e-08", "b20.95_b1_e1e-08", "b20.999_b1_e1e-08"]
SUFFIX = sys.argv[4] if len(sys.argv) > 4 else ""


def sym_sqrt(B):
    w, V = np.linalg.eigh(B)
    return (V * np.sqrt(np.clip(w, 0, None))) @ V.T


def interp_cols(u, table):
    """Linear interpolation of per-column tables (C, G) on the uniform u grid at u (n, C)."""
    du = u_grid[1] - u_grid[0]
    x = np.clip(u / du, 0.0, len(u_grid) - 1.000001)
    k = np.floor(x).astype(np.int64)
    t = x - k
    T = table.T
    return np.take_along_axis(T, k, 0) * (1 - t) + np.take_along_axis(T, k + 1, 0) * t


def tables(t):
    psi, dpsi = r[f"{state}_{t}_psi"], r[f"{state}_{t}_dpsi"]
    safe = np.where(fg > 0, fg, 1.0)
    return np.log(np.where(fg > 0, psi / safe, dpsi)), np.log(dpsi)


B, O = r[f"{state}_B"], r[f"{state}_O"]
noise = draws @ sym_sqrt(B)
tabs = {tag(*cfg): tables(tag(*cfg)) for cfg in CONFIGS}
own = np.empty((n, C))
phi = {t: np.zeros((C, C)) for t in FULL}
t0 = time.time()
for c in range(C):
    q = O[:, c][None, :] + noise
    q -= q.max(1, keepdims=True)
    p = np.exp(q); p /= p.sum(1, keepdims=True)
    own[:, c] = p[:, c]
    af = p.copy(); af[:, c] = 1.0 - p[:, c]          # |f|
    u = np.arcsinh(af / f0)
    for t in FULL:
        ls = tabs[t][1]
        slope = interp_cols(u, ls)
        w = np.exp(slope) * p
        phi[t] += pi[c] * (np.diag(w.mean(0)) - p.T @ w / n)
print(f"{state}: draws and phi done [{time.time() - t0:.0f}s]", flush=True)

out = {"own_mean": own.mean(0)}
for cfg in CONFIGS:
    t = tag(*cfg)
    lr = tabs[t][0]
    a = 1.0 - own                                      # |f_c| for x of class c
    u = np.arcsinh(a / f0)
    psi_abs = a * np.exp(interp_cols(u, lr))
    forcing = MEAN_NORM ** 2 * pi * psi_abs.mean(0)    # = -r^2 chi_cc  (psi odd, f_c < 0)
    out[f"{t}_forcing"] = forcing
    out[f"{t}_restore"] = r[f"{state}_{t}_adam"] - forcing
    if t in FULL:
        out[f"{t}_phiO"] = -np.diag(phi[t].T @ O)
        out[f"{t}_phidiag"] = np.diag(phi[t])
np.savez(OUTDIR + f"/decomp_{state}{SUFFIX}.npz", **out)
print(f"{state}: saved [{time.time() - t0:.0f}s wall, {time.process_time():.0f}s cpu]")
