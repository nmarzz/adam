"""Side checks for the E1a write-up."""
import json, sys
import numpy as np
sys.path.insert(0, __file__.rsplit("/", 1)[0])
from common import *
import analysis as A   # re-runs analysis prints; silence by redirect in caller

r = A.r; pi = A.pi; C = A.C; dec = A.dec
print("\n\n######## CHECKS ########")
# 1. stored total rate: fixed-slope SBPL on classes 1..K vs forcing-based kink
for t in [tag(*c) for c in CONFIGS[:6]]:
    yT = np.abs(r[f"init_{t}_adam"])
    k_force = A.res_kink = None
    out = []
    for K in (40, 64, 128):
        f = A.fit_sbpl(np.arange(K), yT, fixed=(1.0, 0.5), xb0=np.log(predicted_rates(pi, BETA1, *[float(t.split('_')[0][3:]), int(t.split('_')[1][1:]), float(t.split('_e')[1])])['kink_beta2']))
        out.append((K, f["pi_b"], f["rms_log"]))
    kf = A.OUT["kinks"][t]["fixed"]["pi_b"]
    print(f"{t:20s} forcing-fit kink {kf:.3e} | total-rate fixed-slope fits: " +
          "  ".join(f"c<= {K}: {pb:.3e} ({pb / kf:.3f}, rms {rms:.3f})" for K, pb, rms in out))
# 2. restoring fraction among classes <= 64 at init (b=1, beta2=0.99)
for t in ("b20.95_b1_e1e-08", "b20.99_b1_e1e-08", "b20.999_b1_e1e-08", "b20.99_b16_e1e-08"):
    R = r[f"init_{t}_adam"] - dec["init"][f"{t}_forcing"]
    F = dec["init"][f"{t}_forcing"]
    q = np.abs(R / F)
    big = np.flatnonzero(q[:64] > 0.1)
    print(f"{t}: c<=64 with |restore|/forcing>0.1: {(big + 1).tolist()} values {np.round(q[big], 2).tolist()};"
          f" c>64: median {np.median(q[64:]):.2f}, max {q[64:].max():.2f}, frac>0.1 {np.mean(q[64:] > 0.1):.2f};"
          f" negatives {(r[f'init_{t}_adam'] < 0).sum()}")
# 3. PSGD independent of beta2
p95, p99, p999 = (r[f"init_b2{b2}_b1_e1e-08_psgd"] for b2 in ("0.95", "0.99", "0.999"))
print("PSGD max rel diff across beta2 (init):", np.abs(p95 / p99 - 1).max(), np.abs(p999 / p99 - 1).max())
# 4. eps=1e-4 vs eps=1e-8 at b=1 (forcing)
rat = dec["init"]["b20.99_b1_e0.0001_forcing"] / dec["init"]["b20.99_b1_e1e-08_forcing"]
print(f"forcing eps1e-4/eps1e-8 at b=1: min {rat.min():.4f} max {rat.max():.4f} (classes {rat.argmin() + 1}, {rat.argmax() + 1})")
rat16 = dec["init"]["b20.99_b16_e0.0001_forcing"] / dec["init"]["b20.99_b16_e1e-08_forcing"]
print(f"forcing eps1e-4/eps1e-8 at b=16: class1 {rat16[0]:.3f} class10 {rat16[9]:.3f} class128 {rat16[-1]:.3f} min {rat16.min():.3f}")
rat2 = dec["init"]["b20.99_b1_e0.01_forcing"] / dec["init"]["b20.99_b1_e1e-08_forcing"]
print(f"forcing eps1e-2/eps1e-8 at b=1: class1 {rat2[0]:.3f} class10 {rat2[9]:.3f} class128 {rat2[-1]:.3f}")
# 5. eps-dominated rate is b-independent? compare b=1 and b=16 at eps=1e-2 in the deep tail
d1 = dec["init"]["b20.99_b1_e0.01_forcing"]; d16 = dec["init"]["b20.99_b16_e0.01_forcing"]
print(f"eps=1e-2 forcing b16/b1 deep tail (c>=79): median {np.median(d16[78:] / d1[78:]):.3f};"
      f" eps-law r^2 pi/sqrt(eps) ratio for b=16: median {np.median(d16[78:] / (25 * pi[78:] / 0.1)):.3f}")
# 6. Adam/PSGD at the rarest class (forcing)
for t in [tag(*c) for c in CONFIGS[:6]]:
    ratio = dec["init"][f"{t}_forcing"][-1] / abs(r[f"init_{t}_psgd"][-1])
    print(f"{t}: Adam/PSGD at class 128 = {ratio:.3f}  (PSGD/Adam = {1 / ratio:.1f}x)")
# 7. local slope of forcing near pi = 1/C^2 (Adam) vs PSGD
for t in [tag(*c) for c in CONFIGS[:6]]:
    s = np.array(A.OUT["local_slopes"][f"init/{t}/forcing"])
    sp = np.array(A.OUT["local_slopes"][f"init/{t}/psgd"])
    i1 = np.argmin(np.abs(pi - 1.5e-4)); i2 = np.argmin(np.abs(pi - 4e-5))
    print(f"{t}: local slope Adam forcing at pi=1.5e-4: {s[i1]:.3f}, at 4e-5: {s[i2]:.3f} | PSGD: {sp[i1]:.3f}, {sp[i2]:.3f}")
