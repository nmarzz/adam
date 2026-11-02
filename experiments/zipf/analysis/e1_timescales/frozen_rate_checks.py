"""Side checks on rates.npz (C = 128, alpha = 2) used as references in E1b.
Run from the repo root: JAX_PLATFORMS=cpu real/bin/python <this file>. numpy only."""
import numpy as np

from e1b_lib import REPO, ols

rr = dict(np.load(REPO + "/figs/experiments/zipf/rates.npz"))
p = rr["priors"]
lp = np.log(p)

print("1) frozen-rate slopes d log|rate|/d log pi in pi bands (state, config, method)")
bands = [(0.06, 1.0), (0.0143, 0.06), (3e-3, 0.0143), (9e-4, 3e-3), (1.5e-4, 9e-4)]
print("   bands:", bands)
for st in ("init", "mid"):
    for tag in ("b20.99_b1_e1e-08", "b20.95_b1_e1e-08", "b20.99_b16_e1e-08", "b20.99_b1_e0.0001"):
        for m in ("adam", "psgd", "sgd"):
            if m == "sgd" and tag != "b20.99_b1_e1e-08":
                continue
            v = np.abs(rr[f"{st}_{tag}_{m}"])
            out = [ols(lp[(p >= lo) & (p < hi)], np.log(v[(p >= lo) & (p < hi)]))[0] for lo, hi in bands]
            print(f"   {st:4s} {tag:18s} {m:4s}", np.round(out, 3))

print("\n2) eps=1e-4 / eps=1e-8 frozen Adam rate at Theta_0 (b = 1, beta2 = 0.99), per pi band")
a8, a4 = np.abs(rr["init_b20.99_b1_e1e-08_adam"]), np.abs(rr["init_b20.99_b1_e0.0001_adam"])
q = a4 / a8
for lo, hi in [(1e-3, 1), (3e-4, 1e-3), (1.5e-4, 3e-4), (3.7e-5, 1.5e-4)]:
    m = (p >= lo) & (p < hi)
    print(f"   pi in [{lo:.1e},{hi:.1e}) n={m.sum():3d}: mean {q[m].mean():.3f}, sd {q[m].std():.3f}")
mid = np.abs(rr["mid_b20.99_b1_e0.0001_adam"]) / np.abs(rr["mid_b20.99_b1_e1e-08_adam"])
m = (p >= 1.5e-4) & (p <= 1e-3)
print(f"   at the t = 10 Adam state, pi in [1.5e-4, 1e-3]: mean {mid[m].mean():.2f}, range {mid[m].min():.2f}-{mid[m].max():.2f}")

print("\n3) class-to-class scatter of the frozen Adam rate at Theta_0 vs the quoted 4-seed s.e.")
for tag in ("b20.99_b1_e1e-08", "b20.99_b1_e0.0001", "b20.99_b16_e1e-08"):
    a, s = np.abs(rr[f"init_{tag}_adam"]), rr[f"init_{tag}_adam_se"]
    for lo, hi in [(1e-3, 1e-2), (3e-4, 1e-3), (1.5e-4, 3e-4), (3.7e-5, 1.5e-4)]:
        idx = np.flatnonzero((p >= lo) & (p < hi))
        X = np.vander(lp[idx], 3)
        co, *_ = np.linalg.lstsq(X, np.log(a[idx]), rcond=None)
        res = np.log(a[idx]) - X @ co
        print(f"   {tag:18s} pi[{lo:.1e},{hi:.1e}) n={len(idx):2d}: rms resid of log rate about a quadratic"
              f" {res.std():.4f}; median quoted rel s.e. {np.median(s[idx] / a[idx]):.4f}")

print("\n4) is the scatter the random initial overlap O_cc(0)?  (pi in [3.7e-5, 1e-3))")
O = rr["init_O"]
print(f"   O_cc(0): mean {np.diag(O).mean():.4f}, sd {np.diag(O).std():.4f}; off-diagonal sd {O[~np.eye(len(p), dtype=bool)].std():.4f}")
for tag in ("b20.99_b1_e1e-08", "b20.99_b1_e0.0001", "b20.99_b16_e1e-08"):
    a = rr[f"init_{tag}_adam"]
    idx = np.flatnonzero((p >= 3.7e-5) & (p < 1e-3))
    X = np.vander(lp[idx], 3)
    co, *_ = np.linalg.lstsq(X, np.log(np.abs(a[idx])), rcond=None)
    res = np.log(np.abs(a[idx])) - X @ co
    d = np.abs(a[idx]) - np.exp(X @ co)
    o = np.diag(O)[idx]
    k = np.sum(d * o) / np.sum(o ** 2)
    print(f"   {tag:18s}: corr(resid log rate, O_cc(0)) {np.corrcoef(res, o)[0, 1]:+.3f}; d rate / d O_cc {k:.2f};"
          f" fraction of residual variance explained {1 - np.var(d - k * o) / np.var(d):.2f}; all drifts > 0: {bool(np.all(a > 0))}")
