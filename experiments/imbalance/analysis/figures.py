"""Series-I figures A-E (``real/bin/python experiments/imbalance.py plot``).

Colour: Adam's beta2 is ordinal -> one-hue blue ramp (light = short memory); SGD is a
separate entity -> categorical orange; classes by log pi on a sequential blue ramp;
kappa = 0.1 and m1 are dashed (secondary encoding). Steps are the reference palette's
validated values (dataviz skill, references/palette.md).
"""
from __future__ import annotations

import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from matplotlib.lines import Line2D

import lib
from lib import I

sys.path.insert(0, str(lib.REPO))
from experiments.plot_style import apply_paper_style, polish_axis  # noqa: E402

apply_paper_style()
plt.rcParams.update({"font.size": 12, "axes.titlesize": 13, "axes.labelsize": 12,
                     "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 9,
                     "lines.linewidth": 1.6})

BLUE = {0.9: "#6da7ec", 0.95: "#2a78d6", 0.99: "#1c5cab", 0.999: "#0d366b"}
SGD_C = "#eb6834"
INK, MUTED = "#0b0b0b", "#52514e"
CLASS_CMAP = LinearSegmentedColormap.from_list(
    "blues", ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281", "#0d366b"])
DASH = {1.0: "-", 0.1: "--"}
FIG = lib.OUTDIR


def color(opt, b2):
    return BLUE[b2] if opt == "adam" else SGD_C


def opts():
    return [("adam", b2) for b2 in I.BETA2S] + [("sgd", 0.0)]


def name(opt, b2):
    return f"Adam β₂={b2:g}" if opt == "adam" else "SGD"


def save(fig, stem):
    path = FIG / stem
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight", metadata={"CreationDate": None})
    fig.savefig(path.with_suffix(".png"), dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}.{{pdf,png}}")


def risk_of(loss, pi):
    return loss @ pi


def sim_band(ax, sim, c, ls="-", lw=0.0, alpha=0.22, line=False):
    pi, t = sim["priors"], sim["time"]
    R = sim["loss"] @ pi
    m, se = R.mean(0), R.std(0, ddof=1) / np.sqrt(len(R))
    k = t > 0
    ax.fill_between(t[k], (m - 2 * se)[k], (m + 2 * se)[k], color=c, alpha=alpha, lw=0)
    ax.plot(t[k][::6], m[k][::6], "o", color=c, ms=3.5, mfc="none", mew=0.9)
    if line:
        ax.plot(t[k], m[k], color=c, ls=ls)


# --------------------------------------------------------------------------- #
# A: risk vs time
# --------------------------------------------------------------------------- #
def fig_risk():
    for model in I.MODELS:
        for kappa in I.KAPPAS:
            for alpha in I.ALPHAS:
                fig, axes = plt.subplots(2, len(I.CLASSES), figsize=(5 * len(I.CLASSES), 7),
                                         sharex="col",
                                         gridspec_kw={"height_ratios": [2, 1]})
                drawn = False
                for j, C in enumerate(I.CLASSES):
                    ax, ax2 = axes[0, j], axes[1, j]
                    for opt, b2 in opts():
                        c = color(opt, b2)
                        cur = lib.curves(model, opt, kappa, C, alpha, b2)
                        if cur is None:
                            continue
                        drawn = True
                        t, R = cur["time"], risk_of(cur["loss"], cur["priors"])
                        k = t > 0
                        ax.plot(t[k], R[k], color=c, label=name(opt, b2))
                        ax2.plot(t[k], lib.local_exponent(t, R)[k], color=c)
                        sim = lib.load_sim(model, opt, kappa, C, alpha, b2)
                        if sim is not None:
                            sim_band(ax, sim, c)
                    ax.axvspan(1e-2, 1.0, color="0.92", zorder=-1, lw=0)
                    ax.set(xscale="log", yscale="log", title=f"C = {C}", ylabel="risk R(t)")
                    ax2.set(xscale="log", xlabel="t (steps / D)", ylabel="−d log R / d log t",
                            ylim=(-0.2, 1.6))
                    ax2.axhline(0, color=MUTED, lw=0.8)
                    for a in (ax, ax2):
                        polish_axis(a)
                if not drawn:
                    plt.close(fig)
                    continue
                src = "ODE (lines), 16-sim mean (○) ±2 s.e. (band)" if model == "m0" else \
                    "16-run mean ± 2 s.e. (bias model, simulation)"
                fig.suptitle(f"{'no bias' if model == 'm0' else 'with bias'}, K = {kappa:g}·I,"
                             f" α = {alpha:g}: {src}; grey = t < 1", fontsize=12)
                axes[0, 0].legend(frameon=False, loc="lower left")
                fig.tight_layout()
                save(fig, f"A_risk_{model}_k{kappa:g}_a{alpha:g}")

    # overlays (no-bias ODE): K = I vs 0.1 I; base vs tuned eta at each kappa
    for which in ("kappa", "tuned_k1", "tuned_k0.1"):
        fig, axes = plt.subplots(2, len(I.CLASSES), figsize=(5 * len(I.CLASSES), 7),
                                 sharex=True)
        drawn = False
        for i, alpha in enumerate(I.ALPHAS):
            for j, C in enumerate(I.CLASSES):
                ax = axes[i, j]
                for opt, b2 in opts():
                    c = color(opt, b2)
                    if which == "kappa":
                        variants = [(1.0, "base", "-"), (0.1, "base", "--")]
                    else:
                        kap = float(which.split("_k")[1])
                        variants = [(kap, "base", "-"), (kap, "tuned", "--")]
                    for kappa, es, ls in variants:
                        if es == "tuned" and (not I.ETA_FILE.exists()
                                              or f"{kappa:g}" not in I.json.loads(
                                                  I.ETA_FILE.read_text())):
                            continue
                        cur = lib.curves("m0", opt, kappa, C, alpha, b2, etaset=es)
                        if cur is None:
                            continue
                        t, R = cur["time"], risk_of(cur["loss"], cur["priors"])
                        drawn = True
                        k = t > 0
                        ax.plot(t[k], R[k], color=c, ls=ls)
                ax.set(xscale="log", yscale="log", title=f"C = {C}, α = {alpha:g}")
                if i == 1:
                    ax.set_xlabel("t (steps / D)")
                if j == 0:
                    ax.set_ylabel("risk R(t)")
                polish_axis(ax)
        if not drawn:
            plt.close(fig)
            continue
        handles = [Line2D([], [], color=color(o, b), label=name(o, b)) for o, b in opts()]
        handles += ([Line2D([], [], color=INK, ls="-", label="K = I"),
                     Line2D([], [], color=INK, ls="--", label="K = 0.1·I")] if which == "kappa"
                    else [Line2D([], [], color=INK, ls="-", label="E-series η (0.3 / 3)"),
                          Line2D([], [], color=INK, ls="--", label="η tuned for this K")])
        axes[0, 0].legend(handles=handles, frameon=False, loc="lower left")
        fig.suptitle("ODE: K = I vs K = 0.1·I" if which == "kappa"
                     else f"ODE, K = {which.split('_k')[1]}·I: base vs tuned learning rates",
                     fontsize=12)
        fig.tight_layout()
        save(fig, f"A_risk_{which}_overlay")


# --------------------------------------------------------------------------- #
# B: per-class losses
# --------------------------------------------------------------------------- #
def fig_classes():
    for model in I.MODELS:
        for kappa in I.KAPPAS:
            for alpha in I.ALPHAS:
                for C in I.CLASSES:
                    curs = [(o, b, lib.curves(model, o, kappa, C, alpha, b)) for o, b in opts()]
                    curs = [x for x in curs if x[2] is not None]
                    if not curs:
                        continue
                    extra = model == "m1"
                    n = len(curs) + extra
                    fig, axes = plt.subplots(1, n, figsize=(3.3 * n, 3.6), sharey=False)
                    axes = np.atleast_1d(axes)
                    pi = curs[0][2]["priors"]
                    norm = LogNorm(pi.min(), pi.max())
                    for ax, (o, b, cur) in zip(axes, curs):
                        t, L = cur["time"], cur["loss"]
                        k = t > 0
                        for c in range(C)[::-1]:
                            ax.plot(t[k], L[k, c] / L[0, c], color=CLASS_CMAP(norm(pi[c])),
                                    lw=1.0)
                        ax.axhline(0.5, color=MUTED, ls=":", lw=1)
                        ax.set(xscale="log", yscale="log", title=name(o, b),
                               xlabel="t", ylim=(1e-3 if kappa < 1 else 5e-3, 3))
                        polish_axis(ax)
                    axes[0].set_ylabel("ℓ_c(t) / ℓ_c(0)")
                    if extra:
                        ax = axes[-1]
                        lp = np.log(pi) - np.log(pi).mean()
                        for o, b, cur in curs:
                            beta = cur["beta"][:, -1]                 # (runs, C)
                            bc = beta - beta.mean(1, keepdims=True)
                            ax.errorbar(lp, bc.mean(0), yerr=2 * bc.std(0, ddof=1) / np.sqrt(len(bc)),
                                        color=color(o, b), marker="o", ms=4, lw=1,
                                        label=name(o, b))
                        lims = [lp.min() - 0.5, lp.max() + 0.5]
                        ax.plot(lims, lims, color=MUTED, ls=":", lw=1, label="β = log π")
                        ax.set(xlabel="log π_c − mean", ylabel="β_c(T) − mean",
                               title="learned bias at T")
                        ax.legend(frameon=False, fontsize=8)
                        polish_axis(ax)
                    sm = plt.cm.ScalarMappable(norm=norm, cmap=CLASS_CMAP)
                    fig.colorbar(sm, ax=axes[:len(curs)].tolist(), label="π_c", pad=0.01)
                    fig.suptitle(f"{'no bias (ODE)' if model == 'm0' else 'with bias (sim mean)'},"
                                 f" K = {kappa:g}·I, C = {C}, α = {alpha:g}", fontsize=12, y=1.06)
                    save(fig, f"B_classes_{model}_k{kappa:g}_a{alpha:g}_C{C}")


# --------------------------------------------------------------------------- #
# C: per-class rates vs pi
# --------------------------------------------------------------------------- #
def fig_rates():
    for model in I.MODELS:
        for kappa in I.KAPPAS:
            for alpha in I.ALPHAS:
                fig, axes = plt.subplots(3, len(I.CLASSES), figsize=(5 * len(I.CLASSES), 11))
                drawn = False
                for j, C in enumerate(I.CLASSES):
                    ax, ax2, ax3 = axes[:, j]
                    for opt, b2 in opts():
                        cur = lib.curves(model, opt, kappa, C, alpha, b2)
                        if cur is None:
                            continue
                        drawn = True
                        c = color(opt, b2)
                        t, L, pi = cur["time"], cur["loss"], cur["priors"]
                        tc = lib.level_times(t, L, 0.5)
                        ok = np.isfinite(tc)
                        ax.plot(pi[ok], tc[ok], "o-", color=c, ms=4, lw=1, label=name(opt, b2))
                        if (~ok).any():
                            ax.plot(pi[~ok], np.full((~ok).sum(), t[-1]), "^", mfc="none",
                                    color=c, ms=6)
                        s_tc = lib.slope_at(t, L, tc)
                        ax2.plot(pi, s_tc, "o-", color=c, ms=4, lw=1)
                        ax2.plot(pi, lib.late_slopes(t, L), "s", mfc="none", color=c, ms=4)
                        if opt == "adam":
                            for a in (ax, ax2, ax3):
                                a.axvline(lib.kink(b2, C), color=c, ls=":", lw=1)
                    # guides through the SGD and Adam head
                    sg = lib.curves(model, "sgd", kappa, C, alpha)
                    ad = lib.curves(model, "adam", kappa, C, alpha, I.BETA2S[-1])
                    if sg is not None and ad is not None:
                        pi = sg["priors"]
                        tcs = lib.level_times(sg["time"], sg["loss"], 0.5)
                        tca = lib.level_times(ad["time"], ad["loss"], 0.5)
                        x = np.geomspace(pi.min(), pi.max(), 50)
                        ax.plot(x, 0.7 * tcs[0] * (x / pi[0]) ** -1.0, color=MUTED, ls="--",
                                lw=1, label="slope −1")
                        ax.plot(x, 0.7 * tca[0] * (x / pi[0]) ** -0.5, color=MUTED, ls="-.",
                                lw=1, label="slope −½")
                    ax.set(xscale="log", yscale="log", xlabel="π_c", ylabel="t_c (ℓ_c = ℓ_c(0)/2)",
                           title=f"C = {C}")
                    ax2.set(xscale="log", xlabel="π_c", ylabel="−d log ℓ_c/d log t",
                            title="at t_c (filled), last decade (open)")
                    ax2.axhline(0, color=MUTED, lw=0.8)
                    # row 3: frozen drift (m0) or frozen bias drift (m1)
                    if model == "m0":
                        fr = lib.load_frozen(kappa, C, alpha)
                        if fr is not None:
                            pi = fr["priors"]
                            for b2 in I.BETA2S:
                                if f"init_b2{b2:g}_adam" not in fr:
                                    continue
                                ax3.errorbar(pi, np.abs(fr[f"init_b2{b2:g}_adam"]),
                                             yerr=2 * fr[f"init_b2{b2:g}_adam_se"],
                                             color=BLUE[b2], marker="o", ms=4, lw=1)
                            ax3.plot(pi, np.abs(fr[f"init_b2{I.BETA2S[0]:g}_sgd"]), "o-",
                                     color=SGD_C, ms=4, lw=1)
                            ax3.set(ylabel="|dO_cc/dt| / η at Θ₀", yscale="log")
                    else:
                        fb = lib.load_frozen_bias(kappa, C, alpha)
                        if fb is not None:
                            pi = fb["priors"]
                            for b2 in I.BETA2S:
                                ax3.errorbar(pi, fb[f"init_b2{b2:g}_adam"],
                                             yerr=2 * fb[f"init_b2{b2:g}_adam_se"],
                                             color=BLUE[b2], marker="o", ms=4, lw=1)
                            ax3.plot(pi, fb[f"init_b2{I.BETA2S[0]:g}_sgd"], "o-", color=SGD_C,
                                     ms=4, lw=1)
                            ax3.axhline(0, color=MUTED, lw=0.8)
                            ax3.set(ylabel="frozen bias drift (dβ/dt = −η·drift)",
                                    yscale="symlog", )
                            ax3.set_yscale("symlog", linthresh=1e-3)
                    ax3.set(xscale="log", xlabel="π_c", title="frozen drift at Θ₀ (dotted: kink_F)")
                    for a in (ax, ax2, ax3):
                        polish_axis(a)
                if not drawn:
                    plt.close(fig)
                    continue
                axes[0, 0].legend(frameon=False, fontsize=8)
                fig.suptitle(f"{'no bias (ODE)' if model == 'm0' else 'with bias (sim mean)'},"
                             f" K = {kappa:g}·I, α = {alpha:g}; triangles: not learned by T",
                             fontsize=12)
                fig.tight_layout()
                save(fig, f"C_rates_{model}_k{kappa:g}_a{alpha:g}")


# --------------------------------------------------------------------------- #
# D: kink synthesis
# --------------------------------------------------------------------------- #
def fig_kink():
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    a, b, c, d, e, f = axes.ravel()
    drawn = False
    # (a) frozen collapse
    for C, mk in zip(I.CLASSES, ("o", "s", "^")):
        for alpha in I.ALPHAS:
            fr = lib.load_frozen(1.0, C, alpha)
            if fr is None:
                continue
            pi = fr["priors"]
            for b2 in I.BETA2S:
                if f"init_b2{b2:g}_adam" not in fr:
                    continue
                drawn = True
                u = pi / lib.kink(b2, C)
                y = np.abs(fr[f"init_b2{b2:g}_adam"]) / (lib.R ** 2 * np.sqrt(pi))
                a.plot(u, y, mk, color=BLUE[b2], ms=4, mfc="none" if alpha == 1 else BLUE[b2])
    u = np.geomspace(1e-3, 1e2, 100)
    a.plot(u, np.minimum(1, np.sqrt(u)), color=MUTED, ls="--", lw=1, label="min(1, √u)")
    a.set(xscale="log", yscale="log", xlabel="u = π / kink_F",
          ylabel="frozen rate / (r² √π)", title="(a) frozen drift collapse (K = I)")
    a.legend(frameon=False)
    # (b) fitted vs predicted
    for (model, kappa), mk in zip([("m0", 1.0), ("m0", 0.1)], ("o", "o")):
        for C in I.CLASSES:
            for b2 in I.BETA2S:
                kF = lib.kink(b2, C)
                for alpha in I.ALPHAS:
                    rat = ratio_fit(model, kappa, C, alpha, b2)
                    if (rat is not None and rat["interior"] and rat["n_above"] >= 2
                            and rat["n_below"] >= 2):
                        b.plot(kF, rat["pi_b"], mk, color=BLUE[b2], ms=6,
                               mfc="none" if kappa < 1 else BLUE[b2])
            if model == "m0" and kappa == 1.0:
                fz = [lib.load_frozen(1.0, C, al) for al in I.ALPHAS]
                if all(x is not None for x in fz):
                    for b2 in [b for b in I.BETA2S if all(f"init_b2{b:g}_adam" in x for x in fz)]:
                        pis = np.concatenate([x["priors"] for x in fz])
                        rate = np.abs(np.concatenate([x[f"init_b2{b2:g}_adam"] for x in fz]))
                        fit = lib.fit_sbpl(pis, rate, fixed=(1.0, 0.5))
                        b.plot(lib.kink(b2, C), fit["pi_b"], "*", color=BLUE[b2], ms=11,
                               mec=INK, mew=0.5)
    x = np.array([5e-3, 0.3])
    b.plot(x, x, color=MUTED, lw=1)
    b.fill_between(x, x / 2, x * 2, color="0.9", lw=0)
    b.set(xscale="log", yscale="log", xlabel="predicted κ_β(1−β₂)/(bF²)", ylabel="fitted π_b",
          title="(b) kink: ★ frozen, ● ratio fit (open: K = 0.1·I; ≥2 classes/side)")
    # (c) t_A/t_S vs pi at the largest C available (alpha = 2), with predicted asymptotes
    Cc = max([C for C in I.CLASSES if lib.load_ode("m0", "sgd", 1.0, C, 2.0) is not None],
             default=32)
    for alpha in (2.0,):
        for kappa in I.KAPPAS:
            for b2 in I.BETA2S:
                r = adam_sgd_ratio("m0", kappa, Cc, alpha, b2)
                if r is None:
                    continue
                pi, ratio = r
                c.plot(pi, ratio, "o", color=BLUE[b2], ms=4,
                       mfc="none" if kappa < 1 else BLUE[b2])
    for b2 in I.BETA2S:
        for kappa in I.KAPPAS:
            win, rare = lib.ratio_asymptotes(b2, Cc, kappa)
            x = np.geomspace(0.5 * (np.arange(1, Cc + 1) ** -2.0 / np.sum(
                np.arange(1, Cc + 1) ** -2.0)).min(), 0.7, 100)
            c.plot(x, np.maximum(win(x), rare), color=BLUE[b2], ls=DASH[kappa], lw=1)
    c.set(xscale="log", yscale="log", xlabel="π_c", ylabel="t_c(Adam) / t_c(SGD)",
          title=f"(c) C = {Cc}, α = 2: ratio vs zero-parameter prediction")
    # (d) normalised ratio vs u, everything
    for model in I.MODELS:
        for kappa in I.KAPPAS:
            for C in I.CLASSES:
                for alpha in I.ALPHAS:
                    for b2 in I.BETA2S:
                        r = adam_sgd_ratio(model, kappa, C, alpha, b2)
                        if r is None:
                            continue
                        pi, ratio = r
                        win, _ = lib.ratio_asymptotes(b2, C, kappa)
                        d.plot(pi / lib.kink(b2, C), ratio / win(pi),
                               "o" if model == "m0" else "D", color=BLUE[b2], ms=3.5,
                               mfc="none" if kappa < 1 else BLUE[b2], alpha=0.8)
    u = np.geomspace(1e-2, 1e2, 100)
    d.plot(u, np.maximum(1, 1 / np.sqrt(u)), color=MUTED, ls="--", lw=1, label="max(1, u^−½)")
    d.set(xscale="log", yscale="log", xlabel="u = π / kink_F",
          ylabel="(t_A/t_S) / window prediction",
          title="(d) collapse over C, α, β₂ (open: K = 0.1·I)")
    d.legend(frameon=False)
    # (e) beta2 pairs
    for (ba, bb), mk in (((0.9, 0.95), "o"), ((0.95, 0.99), "s"), ((0.99, 0.999), "^")):
        p = lib.coef_beta(bb) / lib.coef_beta(ba)
        for kappa in I.KAPPAS:
            for C in I.CLASSES:
                for alpha in I.ALPHAS:
                    ra = lib.curves("m0", "adam", kappa, C, alpha, ba)
                    rb = lib.curves("m0", "adam", kappa, C, alpha, bb)
                    if ra is None or rb is None:
                        continue
                    pi = ra["priors"]
                    r = (lib.level_times(ra["time"], ra["loss"]) /
                         lib.level_times(rb["time"], rb["loss"]))
                    e.plot(pi, r, mk, color=BLUE[bb], ms=4, mfc="none" if kappa < 1 else BLUE[bb])
        e.axhline(p, color=BLUE[bb], ls="--", lw=1,
                  label=f"t_c({ba:g})/t_c({bb:g}) rare pred {p:.3f}")
    e.axhline(1, color=MUTED, lw=0.8)
    e.set(xscale="log", yscale="log", xlabel="π_c", ylabel="t_c ratio",
          title="(e) β₂ pairs (●: .9/.95, ■: .95/.99, ▲: .99/.999)")
    e.legend(frameon=False, fontsize=8)
    # (f) measured t_A/t_S vs the frozen O_cc-drift and margin-drift ratios (K = I, alpha = 2)
    fz = lib.load_frozen(1.0, Cc, 2.0)
    for b2 in I.BETA2S:
        r = adam_sgd_ratio("m0", 1.0, Cc, 2.0, b2)
        if r is None or fz is None or f"init_b2{b2:g}_margin_adam" not in fz:
            continue
        pi_all = fz["priors"]
        er = I.ETA["sgd"] / I.ETA["adam"]
        fO = er * np.abs(fz[f"init_b2{b2:g}_sgd"]) / np.abs(fz[f"init_b2{b2:g}_adam"])
        fM = er * np.abs(fz[f"init_b2{b2:g}_margin_sgd"]) / np.abs(fz[f"init_b2{b2:g}_margin_adam"])
        f.plot(r[0], r[1], "o", color=BLUE[b2], ms=4)
        f.plot(pi_all, fO, "-", color=BLUE[b2], lw=1)
        f.plot(pi_all, fM, "--", color=BLUE[b2], lw=1)
    f.set(xscale="log", yscale="log", xlabel="π_c", ylabel="t_c(Adam) / t_c(SGD)",
          title=f"(f) C = {Cc}, K = I: measured (●), frozen O_cc (—), margin (--)")
    for ax in axes.ravel():
        polish_axis(ax)
    handles = [Line2D([], [], color=color(o, b), marker="o", ls="", label=name(o, b))
               for o, b in opts()]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    if drawn:
        save(fig, "D_kink")
    else:
        plt.close(fig)


def adam_sgd_ratio(model, kappa, C, alpha, b2):
    a = lib.curves(model, "adam", kappa, C, alpha, b2)
    s = lib.curves(model, "sgd", kappa, C, alpha)
    if a is None or s is None:
        return None
    pi = a["priors"]
    ta, ts = lib.level_times(a["time"], a["loss"]), lib.level_times(s["time"], s["loss"])
    ok = lib.fit_set(pi, ta, a["time"][-1]) & lib.fit_set(pi, ts, s["time"][-1])
    return (pi[ok], ta[ok] / ts[ok]) if ok.any() else None


def ratio_fit(model, kappa, C, alpha, b2):
    r = adam_sgd_ratio(model, kappa, C, alpha, b2)
    if r is None or len(r[0]) < 3:
        return None
    return lib.ratio_break(*r)


# --------------------------------------------------------------------------- #
# E: validation
# --------------------------------------------------------------------------- #
def fig_validation():
    fig, (a, b) = plt.subplots(1, 2, figsize=(13, 4.5))
    labels, x = [], 0
    drawn = False
    for kappa in I.KAPPAS:
        for C in I.CLASSES:
            for alpha in I.ALPHAS:
                for opt, b2 in opts():
                    ode = lib.load_ode("m0", opt, kappa, C, alpha, b2)
                    if ode is None:
                        continue
                    for D, T, mk in ((1000, None, "o"), (2000, 100.0, "^"), (4000, 100.0, "s"),
                                     (4000, 300.0, "s")):
                        sim = lib.load_sim("m0", opt, kappa, C, alpha, b2, D=D, T=T)
                        if sim is None:
                            continue
                        drawn = True
                        pi, t = sim["priors"], sim["time"]
                        Rs = (sim["loss"] @ pi).mean(0)
                        _, lag = lib.time_lag(t, Rs, ode["time"], ode["loss"] @ pi)
                        if len(lag):
                            a.errorbar(x, np.median(lag) - 1,
                                       yerr=[[np.median(lag) - lag.min()], [lag.max() - np.median(lag)]],
                                       fmt=mk, color=color(opt, b2), ms=6, lw=1,
                                       mfc="none" if kappa < 1 else color(opt, b2))
                        tc_s = lib.level_times(t, sim["loss"].mean(0))
                        tc_o = lib.level_times(ode["time"], ode["loss"])
                        ok = np.isfinite(tc_s) & np.isfinite(tc_o) & (tc_o >= 1)
                        b.plot(pi[ok], tc_s[ok] / tc_o[ok], mk, color=color(opt, b2), ms=4,
                               mfc="none" if kappa < 1 else color(opt, b2), alpha=0.8)
                    labels.append(f"κ{kappa:g} C{C} α{alpha:g}" if opt == "adam" and b2 == 0.9
                                  else "")
                    x += 1
    if not drawn:
        plt.close(fig)
        return
    a.axhline(0, color=MUTED, lw=0.8)
    a.set(xlabel="configuration (grouped by κ, C, α)", ylabel="t_sim(R)/t_ode(R) − 1 (median, range)",
          title="(a) time lag at fixed risk, t ≥ 1 (●: D = 1000, ▲: 2000, ■: 4000; open: K = 0.1·I)")
    a.set_xticks(range(len(labels)))
    a.set_xticklabels(labels, rotation=60, fontsize=7, ha="right")
    b.axhline(1, color=MUTED, lw=0.8)
    b.set(xscale="log", yscale="log", xlabel="π_c", ylabel="t_c(sim) / t_c(ODE)",
          title="(b) per-class learning times")
    for ax in (a, b):
        polish_axis(ax)
    handles = [Line2D([], [], color=color(o, bb), marker="o", ls="", label=name(o, bb))
               for o, bb in opts()]
    b.legend(handles=handles, frameon=False, fontsize=8)
    fig.tight_layout()
    save(fig, "E_validation")


def main():
    fig_risk()
    fig_classes()
    fig_rates()
    fig_kink()
    fig_validation()


if __name__ == "__main__":
    main()
