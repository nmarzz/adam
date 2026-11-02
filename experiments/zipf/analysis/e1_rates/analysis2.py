"""E1a extras: background/eps-corrected rare-event prefactor F(a) and corrected kink
prediction; Adam/W collapse in n; restoring fraction at the mid state.

F(a): a class-c event at lag l in a batch of b adds (f xi / b)^2 to S on top of
A = (eps + (1 - pi_c) p_c^2 / b) / (1 - beta2) (floor + softmax background), so
  psi(f) = b c_beta sum_l (beta1/sqrt(beta2))^l g(a / (f^2 beta2^l)),
  g(a) = E[xi^2 / sqrt(xi^2 + a)],  a = b^2 A,
and F = psi(1 - 1/C) / psi_sat with psi_sat = b c_beta sqrt(2/pi) / (1 - beta1/sqrt(beta2)).
Fluctuations of S and the rest of the batch are ignored.  The slope-1 asymptote is
F x (rare-event law) and meets r^2 sqrt(b pi) at  b pi = kappa_beta (1 - beta2) / F^2."""
import json
import sys
import numpy as np

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from common import *

r = load()
pi = r["priors"]
C = len(pi)
res = json.load(open(f"{OUTDIR}/results.json"))
dec = {s: dict(np.load(f"{OUTDIR}/decomp_{s}.npz")) for s in ("init", "mid")}

# Gauss-Hermite (probabilists') nodes for E over xi ~ N(0,1); g has a kink-like
# feature near xi ~ sqrt(a), so use a fine trapezoid grid instead
xi = np.linspace(-9, 9, 360001)
w_xi = np.exp(-xi ** 2 / 2) / np.sqrt(2 * np.pi) * (xi[1] - xi[0])


def g(a):
    a = np.atleast_1d(a)
    return np.array([np.sum(w_xi * xi ** 2 / np.sqrt(xi ** 2 + aa)) for aa in a])


def F_pred(b2, b, e, f=1 - 1 / C, lags=88, pc=1 / C):
    rho = BETA1 / np.sqrt(b2)
    A = (e + pc ** 2 / b) / (1 - b2)
    a = b * b * A
    l = np.arange(lags)
    al = a / (f ** 2 * b2 ** l)
    num = np.sum(rho ** l * g(al))
    den = np.sqrt(2 / np.pi) * np.sum(rho ** l)
    return num / den, a


print("Corrected rare-event prefactor F(a) vs measured (forcing / rare-event law)")
print(f"{'config':20s} {'a_bg':>8s} {'a_eps':>8s} {'F_pred':>7s} | {'meas rarest':>11s} {'meas deep med':>13s} "
      f"{'psi(1)/sat c128':>15s} | {'kink pred':>9s} {'corr pred':>9s} {'meas (fix)':>10s} {'meas/corr':>9s} {'meas/pred':>9s}")
out = {}
for b2, b, e in CONFIGS:
    t = tag(b2, b, e)
    p = predicted_rates(pi, BETA1, b2, e, b)
    Fp, a = F_pred(b2, b, e)
    a_bg = b / (C ** 2 * (1 - b2))
    a_eps = b * b * e / (1 - b2)
    yF = dec["init"][f"{t}_forcing"]
    ratio = yF / p["rare"]
    k = res["kinks"][t]
    kink_corr = kappa_beta(BETA1, b2) * (1 - b2) / (b * Fp ** 2)
    meas = k["fixed"]["pi_b"]
    # psi table directly: psi(1-1/C)/psi_sat for the rarest class
    fg = r["f_grid"]
    psi = r[f"init_{t}_psi"]
    c_beta = (1 - BETA1) / np.sqrt(1 - b2)
    psat = b * c_beta * np.sqrt(2 / np.pi) / (1 - BETA1 / np.sqrt(b2))
    p1 = np.exp(np.interp(np.log(1 - 1 / C), np.log(fg[1:]), np.log(psi[-1, 1:])))
    out[t] = dict(F_pred=float(Fp), a=float(a), a_bg=float(a_bg), a_eps=float(a_eps),
                  F_meas_rarest=float(ratio[-1]), F_meas_deep=float(k["F_rare"]),
                  psi1_over_sat_c128=float(p1 / psat), kink_pred=float(p["kink_beta2"]),
                  kink_corrected=float(kink_corr), kink_meas=float(meas),
                  meas_over_corrected=float(meas / kink_corr), meas_over_pred=float(meas / p["kink_beta2"]))
    print(f"{t:20s} {a_bg:8.4f} {a_eps:8.3g} {Fp:7.3f} | {ratio[-1]:11.3f} {k['F_rare']:13.3f} {p1 / psat:15.3f} | "
          f"{p['kink_beta2']:9.2e} {kink_corr:9.2e} {meas:10.2e} {meas / kink_corr:9.2f} {meas / p['kink_beta2']:9.2f}")

# collapse: n at which Adam/W (forcing part) = 0.5 and 0.8, in units of kappa / F_pred^2
print("\nCollapse of Adam/W (W = r^2 sqrt(b pi)) in n = b pi / (1 - beta2): n(Adam/W = level) / (kappa/F^2)")
coll = {}
for b2, b, e in CONFIGS[:6]:
    t = tag(b2, b, e)
    n = np.array(res["collapse"][t]["n"])
    y = np.array(res["collapse"][t]["adam_over_W"])
    o = np.argsort(n)
    nn, yy = n[o], y[o]
    vals = []
    for lev in (0.3, 0.5, 0.8):
        i = np.flatnonzero((yy[:-1] < lev) & (yy[1:] >= lev))
        if len(i):
            i = i[-1]
            w = (lev - yy[i]) / (yy[i + 1] - yy[i])
            vals.append(float(np.exp(np.log(nn[i]) + w * (np.log(nn[i + 1]) - np.log(nn[i])))))
        else:
            vals.append(None)
    kap = kappa_beta(BETA1, b2)
    Fp = out[t]["F_pred"]
    coll[t] = dict(n_levels=vals, kappa=kap)
    v = [np.nan if x is None else x for x in vals]
    print(f"  {t:20s} n(0.3)={v[0]:.3f} n(0.5)={v[1]:.3f} n(0.8)={v[2]:.3f}"
          f"  | /kappa: {v[0] / kap:.3f} {v[1] / kap:.3f} {v[2] / kap:.3f}"
          f"  | /(kappa/F^2): {v[0] * Fp ** 2 / kap:.3f} {v[1] * Fp ** 2 / kap:.3f} {v[2] * Fp ** 2 / kap:.3f}")

# mid state restoring fraction (median over unlearned tail classes 25-128) and forcing ratio
print("\nMid state: median over classes 25-128 of restoring/forcing and of mid/init forcing; timescales outcome")
learned_by_3000 = {"b20.99_b1_e1e-08": "29/64", "b20.95_b1_e1e-08": "23/64", "b20.99_b16_e1e-08": "64/64",
                   "b20.99_b1_e0.0001": "64/64"}
midout = {}
for b2, b, e in CONFIGS:
    t = tag(b2, b, e)
    F_mid = dec["mid"][f"{t}_forcing"]
    rest = (r[f"mid_{t}_adam"] - F_mid) / F_mid
    tail = slice(24, None)
    fr = F_mid / dec["init"][f"{t}_forcing"]
    midout[t] = dict(restore_median=float(np.median(rest[tail])), restore_min=float(rest[tail].min()),
                     restore_max=float(rest[tail].max()), forcing_ratio_median=float(np.median(fr[tail])),
                     forcing_ratio_max=float(fr[tail].max()))
    print(f"  {t:20s} restoring/forcing median {np.median(rest[tail]):+.3f} [{rest[tail].min():+.3f}, {rest[tail].max():+.3f}]"
          f"  forcing mid/init median {np.median(fr[tail]):.3f} max {fr[tail].max():.3f}"
          f"  {('timescales: ' + learned_by_3000[t] + ' learned by t=3000') if t in learned_by_3000 else ''}")

# phi_cc and O_cc at mid for a few rare classes (b=1 and b=16), equilibrium O* = forcing/phi_cc
print("\nMid state, relaxation picture for rare classes: O* = forcing / phi_cc (diagonal part only)")
O = r["mid_O"]
for t in ("b20.99_b1_e1e-08", "b20.99_b16_e1e-08"):
    ph = dec["mid"][f"{t}_phidiag"]
    F_mid = dec["mid"][f"{t}_forcing"]
    phiO = dec["mid"][f"{t}_phiO"]
    for c in (24, 49, 99, 127):
        diag_part = -ph[c] * O[c, c]
        print(f"  {t:20s} class {c + 1:3d}: forcing {F_mid[c]:.4f}  -(phi^T O)_cc {phiO[c]:+.4f} (diagonal part {diag_part:+.4f})"
              f"  phi_cc {ph[c]:.3f}  O_cc {O[c, c]:.4f}  O* = forcing/phi_cc {F_mid[c] / ph[c]:.4f}"
              f"  relax time 1/(eta phi_cc) = {1 / (0.3 * ph[c]):.1f}")

# phi_cc at init vs mid for the rarest classes (b=1): Adam amplifies the restoring coefficient
print("\nphi_cc (restoring coefficient) for rare classes, init vs mid; SGD value at init ~ (1/C)(1-1/C):")
for t in ("b20.99_b1_e1e-08", "b20.99_b16_e1e-08"):
    pi_i = dec["init"][f"{t}_phidiag"]
    pi_m = dec["mid"][f"{t}_phidiag"]
    print(f"  {t:20s} init: " + " ".join(f"c{c + 1}:{pi_i[c]:.3f}" for c in (24, 49, 99, 127)) +
          "   mid: " + " ".join(f"c{c + 1}:{pi_m[c]:.3f}" for c in (24, 49, 99, 127)) +
          f"   SGD-like (1/C)(1-1/C) = {(1 / C) * (1 - 1 / C):.4f}")

json.dump(dict(F=out, collapse=coll, mid=midout), open(f"{OUTDIR}/results2.json", "w"), indent=1)
print("saved results2.json")
