"""E1b summary figure (PNG in the scratch dir). Run from the repo root with
JAX_PLATFORMS=cpu real/bin/python <this file>. numpy + matplotlib only."""
import numpy as np

from e1b_lib import (ADAM, ETA_ADAM, REPO, SCR, first_crossing, interp_log, kink_beta2, load, ols)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

r = load()
pi, t = r["priors"], r["time"]
C = len(pi)
rates = dict(np.load(REPO + "/figs/experiments/zipf/rates.npz"))

# palette: reference categorical slots 1-4 in fixed order (colour = configuration)
CFG = {
    "adam_b20.99_b1_e1e-08": dict(color="#2a78d6", marker="o", label=r"$\beta_2{=}0.99$, $b{=}1$, $\epsilon{=}10^{-8}$"),
    "adam_b20.95_b1_e1e-08": dict(color="#eb6834", marker="s", label=r"$\beta_2{=}0.95$, $b{=}1$, $\epsilon{=}10^{-8}$"),
    "adam_b20.99_b16_e1e-08": dict(color="#1baf7a", marker="^", label=r"$\beta_2{=}0.99$, $b{=}16$, $\epsilon{=}10^{-8}$"),
    "adam_b20.99_b1_e0.0001": dict(color="#eda100", marker="D", label=r"$\beta_2{=}0.99$, $b{=}1$, $\epsilon{=}10^{-4}$"),
}
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": "#b5b4ae", "axes.linewidth": 0.8,
                     "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.titlesize": 11, "axes.titleweight": "bold", "legend.frameon": False,
                     "legend.fontsize": 8.5, "figure.facecolor": "white", "axes.facecolor": "white"})


def tc(k):
    return r[k + "_tc"]


def local_slope(y_t, half=0.2):
    lp = np.log10(pi)
    out = np.full(C, np.nan)
    fin = np.isfinite(y_t)
    for c in range(C):
        if not fin[c]:
            continue
        m = (np.abs(lp - lp[c]) <= half) & fin
        if m.sum() >= 3:
            out[c] = ols(np.log(pi[m]), np.log(y_t[m]))[0]
        else:
            lo, hi = max(c - 1, 0), min(c + 1, C - 1)
            while hi > c and not fin[hi]:
                hi -= 1
            if hi > lo:
                out[c] = (np.log(y_t[hi]) - np.log(y_t[lo])) / (np.log(pi[hi]) - np.log(pi[lo]))
    return out


def style(ax):
    ax.grid(True, which="major", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


fig, axes = plt.subplots(2, 3, figsize=(17.5, 10.6), constrained_layout=True)
(a, b, c), (d, e, f) = axes

# ---------------- (a) t_c vs pi
for k, st in CFG.items():
    a.plot(pi, tc(k), ls="none", marker=st["marker"], ms=4.5, color=st["color"], mec="white", mew=0.5,
           label="Adam " + st["label"])
a.plot(pi, tc("psgd_b20.99_b1_e1e-08"), "--", color="#2a78d6", lw=1.4, label=r"PSGD $b{=}1$ (any $\beta_2$)")
a.plot(pi, tc("psgd_b20.99_b16_e1e-08"), "--", color="#1baf7a", lw=1.4, label=r"PSGD $b{=}16$")
a.plot(pi, tc("sgd_b1"), "-", color=INK2, lw=1.6, label=r"SGD $b{=}1$")
ct, ctc = r["check_time"], r["check_tc"]
cls = np.arange(27)
med = np.median(ctc[:, cls], 0)
a.errorbar(pi[cls], med, yerr=[med - np.nanmin(ctc[:, cls], 0), np.nanmax(ctc[:, cls], 0) - med], fmt="x",
           color=INK, ms=5, mew=1.1, elinewidth=0.8, capsize=0, label=r"Adam sim. $D{=}4000$ (median, range of 8)")
for k, st in CFG.items():
    if k == "adam_b20.99_b1_e0.0001":
        continue
    b2, bb, _ = ADAM[k]
    a.axvline(kink_beta2(b2, bb), color=st["color"], ls=":", lw=1.3)
x = np.array([1.5e-4, 1.5e-3])
a.plot(x, 0.045 * (x / 1.5e-3) ** -0.5, color=MUTED, lw=1.1)
a.text(1.6e-4, 0.045 * (1.6e-4 / 1.5e-3) ** -0.5 * 0.62, "slope $-1/2$", color=INK2, fontsize=8.5)
a.plot(x, 0.045 * (x / 1.5e-3) ** -1.0, color=MUTED, lw=1.1)
a.text(1.6e-4, 0.045 * (1.6e-4 / 1.5e-3) ** -1 * 1.25, "slope $-1$", color=INK2, fontsize=8.5)
a.text(kink_beta2(.99, 1) * 0.93, 0.04, r"$\kappa_\beta(1-\beta_2)/b$", color=INK2, fontsize=8.5, ha="right")
a.set(xscale="log", yscale="log", xlabel=r"$\pi_c$", ylabel=r"$t_c$  ($\ell_c\leq\ell_c(0)/2$)",
      xlim=(1.2e-4, 0.9), ylim=(0.03, 3000))
a.set_title("(a) learning time per class (dotted: predicted $\\beta_2$ kinks)", loc="left")
a.legend(loc="upper right", fontsize=7.6, ncol=1, handletextpad=0.4, borderaxespad=0.2, frameon=True,
         facecolor="white", edgecolor="none", framealpha=0.9)

# ---------------- (b) local slopes
for k, st in CFG.items():
    ls_ = local_slope(tc(k))
    b.plot(pi, ls_, "-", marker=st["marker"], ms=3.5, lw=1.4, color=st["color"], mec="white", mew=0.4)
b.plot(pi, local_slope(tc("psgd_b20.99_b1_e1e-08")), "--", color="#2a78d6", lw=1.4)
b.plot(pi, local_slope(tc("psgd_b20.99_b16_e1e-08")), "--", color="#1baf7a", lw=1.4)
b.plot(pi, local_slope(tc("sgd_b1")), "-", color=INK2, lw=1.6)
for yv, lab in ((-0.5, "$-1/2$"), (-1.0, "$-1$")):
    b.axhline(yv, color=MUTED, lw=0.9)
    b.text(0.95, yv + 0.04, lab, color=INK2, fontsize=9, ha="right", transform=b.get_yaxis_transform())
for k, st in CFG.items():
    if k == "adam_b20.99_b1_e0.0001":
        continue
    b2, bb, _ = ADAM[k]
    b.axvline(kink_beta2(b2, bb), color=st["color"], ls=":", lw=1.3)
b.text(4.5e-3, -2.92, r"$b{=}1$, $\epsilon{=}10^{-8}$ curves leave the axis (to $-5.6$)"
       "\nand end where classes stop being learned"
       "\n(c$\\geq$30 for $\\beta_2{=}0.99$, c$\\geq$24 for 0.95)", fontsize=8, color=INK2, va="bottom")
b.text(0.62, -0.62, "PSGD (dashed)", fontsize=8.5, color=INK2, ha="right", va="top")
b.text(0.62, -1.08, "SGD", fontsize=8.5, color=INK2, ha="right", va="top")
b.set(xscale="log", xlabel=r"$\pi_c$", ylabel=r"local slope $d\log t_c / d\log\pi_c$ ($\pm0.2$ dec.)",
      xlim=(1.2e-4, 0.9), ylim=(-3.0, -0.35))
b.set_title("(b) local exponent of $t_c(\\pi_c)$", loc="left")

# ---------------- (c) eps = 1e-8 vs eps = 1e-4 control (same frozen drift at Theta_0)
k8, k4 = "adam_b20.99_b1_e1e-08", "adam_b20.99_b1_e0.0001"
ratio = tc(k8) / tc(k4)
m = np.isfinite(ratio)
c.plot(tc(k4)[m], ratio[m], "-o", color="#2a78d6", ms=4, lw=1.4, mec="white", mew=0.4,
       label=r"$t_c(\epsilon{=}10^{-8})/t_c(\epsilon{=}10^{-4})$, class at $t_c(\epsilon{=}10^{-4})$")
B8 = r[k8 + "_Bdiag"][1:, 32:].mean(1)
B4 = r[k4 + "_Bdiag"][1:, 32:].mean(1)
c.plot(t[1:], B8 / B4, "-", color=INK2, lw=1.8, label=r"mean $B_{cc}$ of the 32 rarest columns, $\epsilon{=}10^{-8}$ / $\epsilon{=}10^{-4}$")
s8 = r[k8 + "_coef"][1:, 2, 32:].mean(1)
s4 = r[k4 + "_coef"][1:, 2, 32:].mean(1)
c.plot(t[1:], s8 / s4, ":", color=INK2, lw=1.6, label=r"mean diffusion $\sigma_{cc}$ (same columns), ratio")
c.axhline(1, color=MUTED, lw=0.9)
c.axvspan(14, 25, color="#2a78d6", alpha=0.08, lw=0)
c.text(13, 1.2, "onset $t\\approx$14-25", fontsize=8.5, color=INK2, va="center", ha="right")
c.text(tc(k4)[28] * 0.9, ratio[28], " c=29", fontsize=8, color=INK2, va="center", ha="right")
c.text(tc(k4)[29] * 1.05, 1.0, "c$\\geq$30: never learned with $\\epsilon{=}10^{-8}$", fontsize=8, color=INK2,
       va="bottom", ha="left", rotation=0)
c.set(xscale="log", yscale="log", xlabel="time $t$", ylabel=r"ratio  $\epsilon{=}10^{-8}$ / $\epsilon{=}10^{-4}$",
      xlim=(0.3, 3000), ylim=(0.8, 70))
c.set_title("(c) where Adam ($b{=}1$, $\\epsilon{=}10^{-8}$) leaves its control", loc="left")
c.legend(loc="upper left", fontsize=7.8)

# ---------------- (d) collapse in u = pi / kink
for k, st in CFG.items():
    b2, bb, _ = ADAM[k]
    u = pi / kink_beta2(b2, bb)
    d.plot(u, tc(k) * np.sqrt(bb * pi), "-", marker=st["marker"], ms=3.8, lw=1.2, color=st["color"],
           mec="white", mew=0.4, label="Adam " + st["label"])
u1 = pi / kink_beta2(.99, 1)
u16 = pi / kink_beta2(.99, 16)
d.plot(u1, tc("psgd_b20.99_b1_e1e-08") * np.sqrt(pi), "--", color="#2a78d6", lw=1.2, label=r"PSGD $b{=}1$ (vs $u$ of $\beta_2{=}0.99$)")
d.plot(u16, tc("psgd_b20.99_b16_e1e-08") * np.sqrt(16 * pi), "--", color="#1baf7a", lw=1.2, label=r"PSGD $b{=}16$")
uu = np.geomspace(5e-3, 1e3, 200)
A = 0.30
d.plot(uu, A * np.maximum(1, uu ** -0.5), color=INK, lw=1.0, alpha=0.8, label="theory shape: const, $u^{-1/2}$ below $u{=}1$")
d.axvline(1, color=MUTED, lw=0.9)
d.set(xscale="log", yscale="log", xlabel=r"$u=\pi_c/\pi^*$,  $\pi^*=\kappa_\beta(1-\beta_2)/b$",
      ylabel=r"$t_c\sqrt{b\,\pi_c}$", xlim=(8e-3, 1e3), ylim=(0.1, 30))
d.set_title("(d) collapse: window scaling vs distance to the kink", loc="left")
d.legend(loc="upper right", fontsize=7.6)

# ---------------- (e) Adam / PSGD lag vs u, with the frozen-rate ratio at Theta_0
p128 = rates["priors"]
for k, st in CFG.items():
    b2, bb, eps = ADAM[k]
    tag = k.replace("adam_", "")
    kink = kink_beta2(b2, bb)
    rr_ = tc(k) / tc("psgd_" + tag)
    e.plot(pi / kink, rr_, "-", marker=st["marker"], ms=3.8, lw=1.3, color=st["color"], mec="white", mew=0.4,
           label="Adam " + st["label"])
    fr = np.abs(rates[f"init_{tag}_psgd"]) / np.abs(rates[f"init_{tag}_adam"])
    keep = p128 >= 3e-4
    e.plot(p128[keep] / kink, fr[keep], ":", color=st["color"], lw=1.3)
uu = np.geomspace(5e-3, 1e3, 200)
e.plot(uu, np.maximum(1, uu ** -0.5), color=INK, lw=1.0, alpha=0.8, label=r"sharp: $\max(1,u^{-1/2})$")
e.plot(uu, np.sqrt(1 + 1 / uu), color=INK, lw=1.0, ls="--", alpha=0.6, label=r"quadrature: $\sqrt{1+1/u}$")
e.plot([], [], ":", color=INK2, lw=1.3, label=r"frozen-rate ratio at $\Theta_0$ ($C{=}128$, $\pi\geq3{\cdot}10^{-4}$)")
e.axvline(1, color=MUTED, lw=0.9)
e.set(xscale="log", yscale="log", xlabel=r"$u=\pi_c/\pi^*$", ylabel=r"$t_c(\mathrm{Adam})/t_c(\mathrm{PSGD})$",
      xlim=(8e-3, 1e3), ylim=(0.75, 25))
e.set_title("(e) Adam lags PSGD below the $\\beta_2$ kink", loc="left")
e.legend(loc="upper right", fontsize=7.6)

# ---------------- (f) ODE vs simulation
ode = tc(k8)
cls = np.arange(28)
lr = np.log(ctc[:, cls] / ode[cls])
mean_lr = np.nanmean(lr, 0)
se_lr = np.nanstd(lr, 0, ddof=1) / np.sqrt(np.isfinite(lr).sum(0))
full = np.isfinite(ctc[:, cls]).all(0)
f.axhline(1, color=MUTED, lw=0.9)
f.errorbar(pi[cls][full], np.exp(mean_lr[full]), yerr=[np.exp(mean_lr[full]) - np.exp(mean_lr[full] - se_lr[full]),
           np.exp(mean_lr[full] + se_lr[full]) - np.exp(mean_lr[full])], fmt="o", color="#2a78d6", ms=4.5,
           mec="white", mew=0.5, elinewidth=1.2, capsize=0, label="geometric mean over 8 runs $\\pm$ s.e.")
rmin, rmax = np.nanmin(ctc[:, cls], 0) / ode[cls], np.nanmax(ctc[:, cls], 0) / ode[cls]
f.vlines(pi[cls][full], rmin[full], rmax[full], color=INK2, lw=0.8, alpha=0.7, label="min-max over runs")
f.plot(pi[27], np.nanmedian(ctc[:, 27]) / ode[27], marker="o", mfc="white", mec="#2a78d6", ms=5, ls="none",
       label="c=28: 5/8 runs cross by $t{=}300$")
f.text(0.97, 0.03, "classes 5-27: pooled log-ratio $-0.009\\pm0.006$, $\\sum z^2{=}32$ (23 cl.)\n"
       "classes 1-4: sims 4.3% earlier ($-0.044\\pm0.007$)\n"
       "risk gap for $t\\geq1$: $\\leq0.39\\%$ ($|z|\\leq1.9$)", transform=f.transAxes, ha="right", va="bottom",
       fontsize=8.2, color=INK2)
f.set(xscale="log", yscale="log", xlabel=r"$\pi_c$", ylabel=r"$t_c$(simulation) / $t_c$(ODE)", xlim=(6e-4, 0.9),
      ylim=(0.6, 1.6))
f.set_yticks([0.7, 0.8, 0.9, 1.0, 1.2, 1.4])
f.set_yticklabels(["0.7", "0.8", "0.9", "1", "1.2", "1.4"])
f.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
f.set_title("(f) ODE vs 8 Adam runs ($\\beta_2{=}0.99$, $b{=}1$, $D{=}4000$, $t\\leq300$)", loc="left")
f.legend(loc="upper left", fontsize=7.8)

for ax in axes.flat:
    style(ax)
fig.suptitle(r"E1b: per-class learning times, $C=64$, $\alpha=2$, $\eta_{\rm Adam}=0.3$, $\eta_{\rm SGD}=3$ (ODE, horizon $t=3000$)",
             fontsize=12.5, x=0.01, ha="left", color=INK)
out = SCR + "/e1b_summary.png"
fig.savefig(out, dpi=150)
print("saved", out)
