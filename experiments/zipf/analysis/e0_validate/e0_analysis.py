"""E0 analysis: Adam's warm-up transient (first ~1/(1-beta2) steps, no bias correction) against
the gap after warm-up (t >= 1), its scaling with D, and per-class gaps against D pi_c.
numpy only; run from this directory."""
import glob
import os
import re

import numpy as np

OUT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "figs", "experiments", "zipf"))
WARM = 1000          # steps: 10 / (1 - beta2) for beta2 = 0.99


def total(loss, pri):
    return np.asarray(loss) @ np.asarray(pri)


rows = []
for f in sorted(glob.glob(f"{OUT}/validate_C*_D*.npz")):
    C, a, b, D = re.search(r"C(\d+)_a([\d.]+)_b(\d+)_D(\d+)", f).groups()
    C, a, b, D = int(C), float(a), int(b), int(D)
    d = np.load(f)
    t, pri = d["time"], d["priors"]
    sim, ode = total(d["sim_loss"], pri), total(d["ode_loss"], pri)
    simE, odeE = total(d["sim_err"], pri), total(d["ode_err"], pri)
    runs = sim.shape[0]
    mean, se = sim.mean(0), sim.std(0, ddof=1) / np.sqrt(runs)
    gap = mean - ode
    steps = np.round(t * D)
    early = (steps > 0) & (steps < WARM)
    late = t >= 1.0
    k_e = np.flatnonzero(early)[np.argmax(np.abs(gap[early]))] if early.any() else None
    k_l = np.flatnonzero(late)[np.argmax(np.abs(gap[late]))]
    # time shift: sim ahead by dt  =>  gap ~ dR/dt * dt  (dR/dt < 0)
    dR = np.gradient(ode, t)
    shift = np.where(late & (np.abs(dR) > 0), -gap / np.where(dR == 0, np.nan, dR), np.nan)
    per_run = np.abs(sim - ode[None]).max(1)
    gapE = np.abs(simE.mean(0) - odeE)[late].max()
    rows.append(dict(C=C, a=a, b=b, D=D, early=gap[k_e] if k_e is not None else np.nan,
                     t_early=t[k_e] if k_e is not None else np.nan, late=gap[k_l], t_late=t[k_l],
                     z_late=gap[k_l] / se[k_l], rel_late=gap[k_l] / ode[k_l],
                     shift_med=np.nanmedian(shift[late]), shift_iqr=np.nanpercentile(shift[late], [25, 75]),
                     per_run=np.median(per_run), err_gap=gapE, R_end=ode[-1], T=t[-1]))

print(f"{'case':22s} {'early gap':>10s} {'t':>7s} | {'late gap':>9s} {'t':>6s} {'z':>6s} {'rel':>7s} | "
      f"{'shift dt (median, IQR)':>28s} | {'0-1 gap':>8s} {'R(T)':>7s}")
for r in rows:
    print(f"C{r['C']:<3d} a{r['a']:<4g} b{r['b']:<3d} D{r['D']:<6d} {r['early']:+10.4f} {r['t_early']:7.3f} | "
          f"{r['late']:+9.4f} {r['t_late']:6.1f} {r['z_late']:+6.1f} {r['rel_late']:+7.2%} | "
          f"{r['shift_med']:+9.4f} [{r['shift_iqr'][0]:+.4f},{r['shift_iqr'][1]:+.4f}] | "
          f"{r['err_gap']:8.4f} {r['R_end']:7.4f}")

print("\nD-scaling (log-log slope of |gap| vs D) for D-sweep cases:")
keys = sorted({(r["C"], r["a"], r["b"]) for r in rows})
for key in keys:
    sel = sorted([r for r in rows if (r["C"], r["a"], r["b"]) == key], key=lambda r: r["D"])
    if len(sel) < 3:
        continue
    Ds = np.array([r["D"] for r in sel], float)
    for name in ("early", "late", "shift_med"):
        v = np.abs(np.array([r[name] for r in sel]))
        slope = np.polyfit(np.log(Ds), np.log(v), 1)[0]
        print(f"  C{key[0]} a{key[1]:g} b{key[2]}: {name:9s} " +
              " ".join(f"D{int(D)}:{x:.4f}" for D, x in zip(Ds, v)) + f"   slope {slope:+.2f}")

print("\nPer-class late gaps (t >= 1): |mean sim ell_c - ODE ell_c| relative to ODE ell_c, vs D pi_c")
allx, ally, allz = [], [], []
for f in sorted(glob.glob(f"{OUT}/validate_C*_D*.npz")):
    C, a, b, D = re.search(r"C(\d+)_a([\d.]+)_b(\d+)_D(\d+)", f).groups()
    C, a, b, D = int(C), float(a), int(b), int(D)
    d = np.load(f)
    t, pri = d["time"], d["priors"]
    late = t >= 1.0
    sim, ode = d["sim_loss"][:, late], d["ode_loss"][late]
    mean, se = sim.mean(0), sim.std(0, ddof=1) / np.sqrt(sim.shape[0])
    rel = np.abs(mean - ode) / ode            # (T, C)
    z = np.abs(mean - ode) / np.maximum(se, 1e-12)
    k = rel.argmax(0)
    worst = np.array([rel[k[c], c] for c in range(C)])
    zw = np.array([z[k[c], c] for c in range(C)])
    allx += list(D * pri); ally += list(worst); allz += list(zw)
    print(f"C{C:<3d} a{a:<4g} b{b:<3d} D{D:<6d} worst rel gap: class1 {worst[0]:.3f} (z {zw[0]:.1f}), "
          f"median class {np.median(worst):.3f}, rarest {worst[-1]:.3f} (z {zw[-1]:.1f}, D pi_C = {D * pri[-1]:.1f})")
allx, ally, allz = map(np.asarray, (allx, ally, allz))
for lo, hi in ((0, 10), (10, 100), (100, 1000), (1000, 1e9)):
    m = (allx >= lo) & (allx < hi)
    if m.any():
        print(f"  D pi_c in [{lo:g}, {hi:g}): n={m.sum():4d}  median rel gap {np.median(ally[m]):.3f}, "
              f"90th pct {np.percentile(ally[m], 90):.3f}, median z {np.median(allz[m]):.1f}")
