"""E1a summary figure (PNG) from rates.npz, decomp_*.npz, results*.json."""
import json
import os
import sys
import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/tmp/adam-matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from common import *

r = load()
pi = r["priors"]
C = len(pi)
R2 = MEAN_NORM ** 2
res = json.load(open(f"{OUTDIR}/results.json"))
res2 = json.load(open(f"{OUTDIR}/results2.json"))
dec = {s: dict(np.load(f"{OUTDIR}/decomp_{s}.npz")) for s in ("init", "mid")}

# categorical palette (fixed order), text inks
PAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#d9d8d4"
MK = ["o", "s", "^", "D", "v", "P"]
plt.rcParams.update({"font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.titlesize": 11, "legend.fontsize": 8.5, "legend.frameon": False,
                     "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": "white"})


def grid(ax):
    ax.grid(True, which="major", color=GRID, lw=0.6)
    ax.set_axisbelow(True)


fig, axes = plt.subplots(2, 3, figsize=(17, 10.5), constrained_layout=True)

# (a) collapse of Adam/W vs n / (kappa / F^2) -----------------------------------
ax = axes[0, 0]
labels = {"b20.95_b1_e1e-08": r"$\beta_2=0.95,\ b=1$", "b20.99_b1_e1e-08": r"$\beta_2=0.99,\ b=1$",
          "b20.999_b1_e1e-08": r"$\beta_2=0.999,\ b=1$", "b20.99_b4_e1e-08": r"$\beta_2=0.99,\ b=4$",
          "b20.99_b16_e1e-08": r"$\beta_2=0.99,\ b=16$", "b20.99_b64_e1e-08": r"$\beta_2=0.99,\ b=64$"}
for i, (b2, b, e) in enumerate(CONFIGS[:6]):
    t = tag(b2, b, e)
    Fp = res2["F"][t]["F_pred"]
    x = b * pi / (kappa_beta(BETA1, b2) * (1 - b2)) * Fp ** 2
    y = dec["init"][f"{t}_forcing"] / (R2 * np.sqrt(b * pi))
    ax.plot(x, y, MK[i], color=PAL[i], ms=4.5, mfc="none" if i % 2 else PAL[i], mew=1.1, label=labels[t])
xx = np.geomspace(1e-4, 1e4, 50)
ax.plot(xx, np.ones_like(xx), "-", color=INK2, lw=1.2)
ax.plot(xx, np.sqrt(xx), "--", color=INK2, lw=1.2)
ax.text(2e2, 1.12, r"window: $r^2\sqrt{b\pi}$ (slope 1/2)", color=INK2, fontsize=8.5, ha="center")
ax.text(2.5e-2, 0.03, "rare events:\n" r"$F(a)\times$ rare-event law (slope 1)", color=INK2, fontsize=8.5)
ax.axvline(1, color=INK2, lw=0.8, ls=":")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlim(3e-4, 5e3); ax.set_ylim(0.012, 1.6)
ax.set_xlabel(r"$b\pi_c\,F(a)^2/(\kappa_\beta(1-\beta_2))$  (corrected kink at 1)")
ax.set_ylabel(r"Adam rate / $r^2\sqrt{b\pi_c}$   (forcing part, $\Theta_0$)")
ax.set_title("(a) Collapse of the frozen rates, $\\epsilon=10^{-8}$")
ax.legend(loc="lower right", ncol=1)
grid(ax)

# (b) measured / predicted kink --------------------------------------------------
ax = axes[0, 1]
names, vs_h1, vs_corr = [], [], []
short = {"b20.95_b1_e1e-08": "β2 .95", "b20.99_b1_e1e-08": "β2 .99", "b20.999_b1_e1e-08": "β2 .999",
         "b20.99_b4_e1e-08": "b 4", "b20.99_b16_e1e-08": "b 16", "b20.99_b64_e1e-08": "b 64",
         "b20.99_b1_e0.0001": "b1 ε1e-4", "b20.99_b1_e0.01": "b1 ε1e-2",
         "b20.99_b16_e0.0001": "b16 ε1e-4", "b20.99_b16_e0.01": "b16 ε1e-2"}
for b2, b, e in CONFIGS:
    t = tag(b2, b, e)
    k = res["kinks"][t]
    meas = k["fixed"]["pi_b"]
    # H1 prediction: beta2 kink for eps=1e-8, else the larger of the two H1 kinks
    h1 = max(k["pred_beta2"], k["pred_eps"])
    names.append(short[t]); vs_h1.append(meas / h1); vs_corr.append(res2["F"][t]["meas_over_corrected"])
xi = np.arange(len(names))
ax.axhline(1, color=INK2, lw=0.8)
ax.plot(xi, vs_h1, "o", color=PAL[1], ms=8, label=r"vs H1: $\max(\kappa_\beta(1-\beta_2)/b,\ b\epsilon)$")
ax.plot(xi, vs_corr, "s", color=PAL[0], ms=7, label=r"vs corrected: $\kappa_\beta(1-\beta_2)/(bF(a)^2)$")
for i, v in enumerate(vs_h1):
    ax.annotate(f"{v:.2f}", (i, v), textcoords="offset points", xytext=(6, 4), fontsize=8, color=INK)
ax.set_yscale("log")
ax.set_xticks(xi); ax.set_xticklabels(names, rotation=40, ha="right")
ax.set_ylabel(r"measured kink / predicted kink")
ax.set_title("(b) Kink position at $\\Theta_0$ (SBPL asymptote intersection)")
ax.legend(loc="upper left")
ax.set_ylim(0.85, 2.6)
ax.text(8, 1.3, "b1 ε1e-4: H1 also predicts a kink\nat bε = 1e-4; none is seen\n(measured kink = 150 bε)",
        fontsize=8, color=INK2, ha="center")
grid(ax)

# (c) eps sweep: Adam/W vs pi -----------------------------------------------------
ax = axes[0, 2]
cases = [("b20.99_b1_e1e-08", PAL[0], "-", r"$b=1,\ \epsilon=10^{-8}$"),
         ("b20.99_b1_e0.0001", PAL[0], "--", r"$b=1,\ \epsilon=10^{-4}$"),
         ("b20.99_b1_e0.01", PAL[0], ":", r"$b=1,\ \epsilon=10^{-2}$"),
         ("b20.99_b16_e1e-08", PAL[1], "-", r"$b=16,\ \epsilon=10^{-8}$"),
         ("b20.99_b16_e0.0001", PAL[1], "--", r"$b=16,\ \epsilon=10^{-4}$"),
         ("b20.99_b16_e0.01", PAL[1], ":", r"$b=16,\ \epsilon=10^{-2}$")]
for t, col, ls, lab in cases:
    b = 16 if "_b16_" in t else 1
    y = dec["init"][f"{t}_forcing"] / (R2 * np.sqrt(b * pi))
    ax.plot(pi, y, ls, color=col, lw=1.8, label=lab)
    kb = res["kinks"][t]["fixed"]["pi_b"]
    if kb < pi.max():
        ax.plot(kb, np.exp(np.interp(np.log(kb), np.log(pi[::-1]), np.log(y[::-1]))), "o", color=col,
                ms=7, mfc="white", mew=1.8)
for b, col in ((1, PAL[0]), (16, PAL[1])):
    for e in (1e-4, 1e-2):
        if pi.min() < b * e < pi.max():
            ax.axvline(b * e, color=col, lw=1.2, ls=(0, (2, 2)), ymax=0.07)
ax.text(0.03, 0.97, "open circles: measured kinks (SBPL)\nshort lines: H1 ε kink at π = bε", transform=ax.transAxes,
        fontsize=8.5, color=INK2, va="top")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_ylim(0.01, 2.0)
ax.set_xlabel(r"$\pi_c$")
ax.set_ylabel(r"Adam rate / $r^2\sqrt{b\pi_c}$  (forcing, $\Theta_0$, $\beta_2=0.99$)")
ax.set_title("(c) $\\epsilon$ sweep: the $\\epsilon$ kink needs $b>b^*=\\sqrt{(1-\\beta_2)/\\epsilon}$")
ax.legend(loc="lower right", ncol=2, bbox_to_anchor=(1.0, 0.07))
grid(ax)

# (d) Theta_0 tail: stored rate vs forcing, sim check ------------------------------
ax = axes[1, 0]
t = "b20.99_b1_e1e-08"
a = r[f"init_{t}_adam"]
F = dec["init"][f"{t}_forcing"]
p = predicted_rates(pi, BETA1, 0.99, 1e-8, 1)
pos = a > 0
ax.plot(pi[pos], a[pos], "o", color=PAL[0], ms=4, label="stored Adam rate (finite-$D$ $\\Theta_0$)")
ax.plot(pi[~pos], -a[~pos], "x", color=PAL[7], ms=6, mew=1.5, label="stored rate < 0 (|value|)")
ax.plot(pi, F, "-", color=INK, lw=1.4, label=r"forcing $r^2\pi_c\,\mathbb{E}\psi_c(1-p_c)$ ($O=0$)")
ax.plot(pi, np.abs(r[f"init_{t}_psgd"]), "-", color=PAL[2], lw=1.4, label="PSGD")
ax.plot(pi, p["rare"], "--", color=PAL[1], lw=1.2, label="rare-event law (H1)")
ax.plot(pi, np.abs(r[f"init_{t}_sgd"]), "-", color=INK2, lw=1, alpha=0.7, label="SGD")
cc = r["check_classes"]
ax.errorbar(pi[cc], r["check_rate"], yerr=r["check_se"], fmt="D", color=PAL[6], ms=7, mfc="none", mew=1.5,
            label="direct frozen simulation")
ax.axvline(res["kinks"][t]["fixed"]["pi_b"], color=INK2, lw=0.8, ls=":")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_ylim(5e-4, 60)
ax.set_xlabel(r"$\pi_c$")
ax.set_ylabel(r"$dO_{cc}/dt\,/\,\eta$")
ax.set_title("(d) $\\Theta_0$, $\\beta_2=0.99$, $b=1$: tail scatter is $-(\\phi^\\top O)_{cc}$")
ax.legend(loc="upper left", fontsize=8)
grid(ax)

# (e) mid vs init -------------------------------------------------------------------
ax = axes[1, 1]
for (t, col, lab) in (("b20.99_b1_e1e-08", PAL[0], "b=1"), ("b20.99_b16_e1e-08", PAL[1], "b=16"),
                      ("b20.95_b1_e1e-08", PAL[2], r"$\beta_2=0.95$, b=1")):
    ax.plot(pi, dec["init"][f"{t}_forcing"], "-", color=col, lw=1.0, alpha=0.6)
    ax.plot(pi, np.abs(r[f"mid_{t}_adam"]), "o", color=col, ms=3.5, label=f"{lab}: total at t=10")
    ax.plot(pi, dec["mid"][f"{t}_forcing"], "--", color=col, lw=1.4)
ax.plot([], [], "-", color=INK2, lw=1.0, alpha=0.6, label=r"forcing at $\Theta_0$")
ax.plot([], [], "--", color=INK2, lw=1.4, label="forcing at t=10")
learned = np.array(res["mid"]["learned"]) - 1
ax.axvspan(pi[learned].min(), pi.max(), color=GRID, alpha=0.5, lw=0)
ax.text(pi[learned].min() * 1.1, 40, r"learned by $t=10$" "\n" r"($\ell_c\leq\ell_c(0)/2$)", fontsize=8.5, color=INK2)
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel(r"$\pi_c$")
ax.set_ylabel(r"$|dO_{cc}/dt|\,/\,\eta$")
ax.set_title("(f) Mid state ($t=10$): head learned, tail forcing unchanged,\nrestoring term cancels 50% (b=1)",
             fontsize=10.5)
ax.set_title("(e) Mid state ($t=10$, $\\beta_2=0.99$ b=1 trajectory)")
ax.legend(loc="upper left", fontsize=8)
grid(ax)

# (f) psi curves ----------------------------------------------------------------------
ax = axes[1, 2]
fg = r["f_grid"]
for (t, ls, lab) in (("b20.99_b1_e1e-08", "-", "b=1"), ("b20.99_b16_e1e-08", "--", "b=16"),
                     ("b20.99_b16_e0.01", ":", r"b=16, $\epsilon=10^{-2}$")):
    b = 16 if "_b16_" in t else 1
    c_beta = (1 - BETA1) / np.sqrt(1 - 0.99)
    psat = b * c_beta * np.sqrt(2 / np.pi) / (1 - BETA1 / np.sqrt(0.99))
    psi = r[f"init_{t}_psi"]
    for j, c in enumerate([0, 9, 49, 127]):
        ax.plot(fg[1:], psi[c, 1:] / psat, ls, color=PAL[j], lw=1.6,
                label=(f"class {c + 1} ($\\pi$={pi[c]:.1e})" if t == "b20.99_b1_e1e-08" else None))
ax.plot([], [], "-", color=INK2, label="b=1"); ax.plot([], [], "--", color=INK2, label="b=16")
ax.plot([], [], ":", color=INK2, label=r"b=16, $\epsilon=10^{-2}$")
ax.axvline(1 - 1 / C, color=INK2, lw=0.8)
ax.axhline(1, color=INK2, lw=0.8, ls=":")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlim(1e-4, 1.3); ax.set_ylim(1e-4, 3)
ax.set_xlabel(r"$|f|$  (vertical line: own-class error $1-1/C$ at $\Theta_0$)")
ax.set_ylabel(r"$\psi_c(|f|)\,/\,\psi_{sat}$")
ax.text(1.2e-4, 1.2, r"$\psi_{sat}=b\,c_\beta\sqrt{2/\pi}/(1-\beta_1/\sqrt{\beta_2})$", fontsize=8, color=INK2)
ax.set_title("(f) $\\psi_c$ at $\\Theta_0$, $\\beta_2=0.99$: linear, saturated, $\\epsilon$ floor")
ax.legend(loc="lower right", fontsize=8, ncol=1)
grid(ax)

fig.suptitle("E1a frozen per-class rates of $O_{cc}$, C = 128, $\\alpha=2$ (rates.npz)", fontsize=13, color=INK)
fig.savefig(f"{OUTDIR}/e1a_rates_summary.png", dpi=110)
print("saved", f"{OUTDIR}/e1a_rates_summary.png")
