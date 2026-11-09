"""Series J Stage 4 figures for S3a (alpha = 0). Run on CPU:
``JAX_PLATFORMS=cpu real/bin/python experiments/knee/analysis/figures_knee.py``.

J1 per-class loss panels (adaptive y-limits)     J2 cliff linearity (ln ell against linear t)
J3 collapse onto L_C(m, s^2)                     J4 knee margin and loss against r'
J5 normaliser regime map over time               J6 K2 balance ratio over time
J7 late loss against the optimum ell*            J8 frozen field: P1 slopes and P6 split
Colour: beta2 lanes on one blue ramp (light = short memory), SGD orange; eps = 1e-16 dashed.
"""
from __future__ import annotations

import json
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

import lib_knee as K

sys.path.insert(0, str(K.REPO))
from experiments.plot_style import apply_paper_style, polish_axis  # noqa: E402

apply_paper_style()
plt.rcParams.update({"font.size": 11, "axes.titlesize": 11, "axes.labelsize": 11,
                     "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 8,
                     "lines.linewidth": 1.4})
BLUE = {0.9: "#6da7ec", 0.95: "#2a78d6", 0.99: "#1c5cab", 0.999: "#0d366b"}
SGD_C, INK, MUTED = "#eb6834", "#0b0b0b", "#52514e"
OUT = K.OUTDIR / "figures"
OUT.mkdir(parents=True, exist_ok=True)
NUM = json.loads((K.OUTDIR / "s3a_numbers.json").read_text())


def save(fig, stem):
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight", metadata={"CreationDate": None})
    fig.savefig(OUT / f"{stem}.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {OUT / stem}.{{pdf,png}}")


def lanes(C, kappa):
    """[(label, colour, group, lane index, dashed)] for the Adam eps arms and SGD."""
    out = []
    for eps, dashed in ((1e-8, False), (1e-16, True)):
        g = K.find("adam", C, kappa, eps)
        if g:
            d = K.load(g["name"])
            for i, b in enumerate(d["beta2s"]):
                out.append((f"{b:g}", BLUE[float(b)], g, i, dashed))
    g = K.find("sgd", C, kappa)
    if g:
        out.append(("sgd", SGD_C, g, 0, False))
    return out


def mk(g, lane_label):
    e = f"_e{g['eps']:g}" if g["opt"] == "adam" else ""
    return NUM["markers"].get(f"C{g['C']}_k{g['kappa']:g}{e}_D{g['D']}_{lane_label}", {})


# --------------------------------------------------------------------------- #
def fig1():
    for C in (128, 32):
        cols = [f"{b:g}" for b in K.BETA2S] + ["sgd"]
        fig, axes = plt.subplots(len(K.KAPPAS), len(cols), figsize=(3.0 * len(cols), 2.5 * len(K.KAPPAS)),
                                 sharex=True)
        for i, kappa in enumerate(K.KAPPAS):
            lo = 1.0
            for lab, col, g, lane, dashed in lanes(C, kappa):
                ax = axes[i, cols.index(lab)]
                c = K.lane_curves(K.load(g["name"]), lane)
                t, rel = c["t"], c["ell_c"] / c["ell_c"][0]
                k = t > 0
                if not dashed:
                    ax.fill_between(t[k], rel[k].min(1), rel[k].max(1), color=col, alpha=0.25, lw=0)
                ax.plot(t[k], (c["ell"] / c["ell"][0])[k], color=col, ls="--" if dashed else "-")
                m = mk(g, lab)
                for tm, sym in ((m.get("t_c"), "o"), (m.get("t_knee"), "x")):
                    if tm is not None and np.isfinite(tm):
                        ax.plot(tm, np.interp(np.log(tm), np.log(t[k]), (c["ell"] / c["ell"][0])[k]),
                                sym, color=INK, ms=4)
                lo = min(lo, np.nanmin(c["ell"][k] / c["ell"][0]))
            opt = K.optimum(C, kappa)["ell_rel"]
            for j in range(len(cols)):
                ax = axes[i, j]
                if opt > lo / 10:
                    ax.axhline(opt, color=MUTED, ls=":", lw=1)
                ax.set(xscale="log", yscale="log", ylim=(max(lo / 3, 1e-14), 2))
                if i == 0:
                    ax.set_title(f"β₂ = {cols[j]}" if cols[j] != "sgd" else "SGD")
                if j == 0:
                    ax.set_ylabel(f"κ = {kappa:g}\nℓ/ℓ₀")
                if i == len(K.KAPPAS) - 1:
                    ax.set_xlabel("t")
                polish_axis(ax)
        handles = [Line2D([], [], color=INK, ls="-", label="ε = 1e-8 (band: class range)"),
                   Line2D([], [], color=INK, ls="--", label="ε = 1e-16"),
                   Line2D([], [], color=INK, marker="o", ls="", label="t_c"),
                   Line2D([], [], color=INK, marker="x", ls="", label="knee"),
                   Line2D([], [], color=MUTED, ls=":", label="optimum ℓ*")]
        fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False)
        fig.suptitle(f"S3a: per-class loss, C = {C}, α = 0 (seed means of 32 runs)", y=1.0)
        fig.tight_layout(rect=(0, 0.03, 1, 0.99))
        save(fig, f"J1_class_loss_C{C}")


def fig2():
    fig, axes = plt.subplots(2, len(K.KAPPAS), figsize=(3.2 * len(K.KAPPAS), 6.5))
    for r, C in enumerate((128, 32)):
        for j, kappa in enumerate(K.KAPPAS):
            ax = axes[r, j]
            tk = []
            for lab, col, g, lane, dashed in lanes(C, kappa):
                if dashed:
                    continue
                c = K.lane_curves(K.load(g["name"]), lane)
                m = mk(g, lab)
                tmax = 3 * (m.get("t_knee") or c["t"][-1])
                k = c["t"] <= min(tmax, c["t"][-1])
                ax.plot(c["t"][k], np.log(c["ell"][k] / c["ell"][0]), color=col, label=lab)
                tk.append(tmax)
            ax.set(title=f"C = {C}, κ = {kappa:g}", xlabel="t (linear)",
                   ylabel="ln(ℓ/ℓ₀)" if j == 0 else None)
            polish_axis(ax)
    axes[0, 0].legend(frameon=False)
    fig.suptitle("J2: cliff linearity — ln ℓ against linear t up to 3 t_knee (ε = 1e-8)")
    fig.tight_layout()
    save(fig, "J2_cliff_linearity")


def fig3():
    sys.path.insert(0, str(K.HERE))
    import diagnostics as D
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    for ax, C in zip(axes, (128, 32)):
        tables = {}
        for f in sorted(K.FIELD.glob(f"J_*_C{C}_a0_*.npz")):
            z = np.load(f)
            kappa = float(z["kappa"])
            m, s2, ell = z["m"].mean(-1), z["s2"].mean(-1), z["loss_state"].mean(-1)
            if kappa not in tables:
                tables[kappa] = D.ShapeTable(C, float(np.sqrt(np.nanmax(z["s2"])) * 1.05 + 0.1),
                                             -1.0, float(np.nanmax(z["m"]) + 2))
            Lc = tables[kappa](m, s2)
            opt = str(z["opt"])
            for lane in range(m.shape[1]):
                col = SGD_C if opt == "sgd" else BLUE[float(z["beta2s"][lane])]
                ax.plot(Lc[:, lane], ell[:, lane], "o", color=col, ms=2.5, alpha=0.7)
        x = np.array([1e-12, 10])
        ax.plot(x, x, color=MUTED, lw=1)
        ax.set(xscale="log", yscale="log", xlabel="L_C(m, s²) at the seed-mean state",
               ylabel="MC class loss at that state", title=f"C = {C}, all κ, ε = 1e-8")
        polish_axis(ax)
    handles = [Line2D([], [], color=BLUE[b], marker="o", ls="", label=f"Adam β₂={b:g}") for b in K.BETA2S]
    handles.append(Line2D([], [], color=SGD_C, marker="o", ls="", label="SGD"))
    axes[0].legend(handles=handles, frameon=False)
    fig.suptitle("J3: collapse onto the exact shape function L_C(m, s²)")
    fig.tight_layout()
    save(fig, "J3_shape_collapse")


def fig4():
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    for C, mkr in ((128, "o"), (32, "s")):
        for eps, fill in ((1e-8, True), (1e-16, False)):
            for b in K.BETA2S:
                rp, mm, ll = [], [], []
                for kappa in K.KAPPAS:
                    g = K.find("adam", C, kappa, eps)
                    m = mk(g, f"{b:g}") if g else {}
                    if m.get("m_knee") is not None:
                        rp.append(K.R / np.sqrt(kappa))
                        mm.append(m["m_knee"])
                        ll.append(m["ell_knee_rel"])
                axes[0].plot(rp, mm, mkr + "-", color=BLUE[b], mfc=BLUE[b] if fill else "none",
                             ls="-" if fill else "--", ms=5, lw=0.8)
                axes[1].plot(rp, ll, mkr + "-", color=BLUE[b], mfc=BLUE[b] if fill else "none",
                             ls="-" if fill else "--", ms=5, lw=0.8)
    rp = K.R / np.sqrt(np.array(K.KAPPAS))
    axes[0].plot(rp, 1.3 * rp, color=MUTED, ls=":", label="∝ r'")
    axes[0].plot(rp, 0.06 * rp ** 2, color=MUTED, ls="-.", label="∝ r'²")
    axes[0].set(xscale="log", yscale="log", xlabel="r' = 5/√κ", ylabel="margin at the knee m_knee")
    axes[1].set(xscale="log", yscale="log", xlabel="r' = 5/√κ", ylabel="ℓ_knee / ℓ₀")
    axes[0].legend(frameon=False)
    for ax in axes:
        polish_axis(ax)
    fig.suptitle("J4: knee position against noise (●: C = 128, ■: C = 32; filled ε = 1e-8, open ε = 1e-16)")
    fig.tight_layout()
    save(fig, "J4_knee_vs_r")


def fig5():
    from matplotlib.colors import ListedColormap
    cmap = ListedColormap(["#eb6834", "#86b6ef", "#104281"])        # eps, background, own
    for C in (128, 32):
        rows, labels, knees = [], [], []
        tedges = np.geomspace(0.05, 3000, 201)
        tgrid = np.sqrt(tedges[:-1] * tedges[1:])
        for kappa in K.KAPPAS:
            f = K.FIELD / f"{K.find('adam', C, kappa, 1e-8)['name']}.npz"
            if not f.exists():
                continue
            z = np.load(f)
            for lane, b in enumerate(z["beta2s"]):
                code = z["norm_code"][:, lane]
                maj = np.array([np.bincount(c[np.isfinite(c)].astype(int), minlength=3).argmax()
                                if np.isfinite(c).any() else -1 for c in code])
                row = np.full(len(tgrid), np.nan)
                inside = (tgrid >= z["time"][0]) & (tgrid <= z["time"][-1])
                idx = np.searchsorted(z["time"], tgrid[inside], side="right") - 1
                row[inside] = maj[np.clip(idx, 0, len(maj) - 1)]
                rows.append(row)
                labels.append(f"κ={kappa:g} β₂={b:g}")
                g = K.find("adam", C, kappa, 1e-8)
                knees.append(mk(g, f"{b:g}").get("t_knee"))
        if not rows:
            continue
        fig, ax = plt.subplots(figsize=(10, 0.28 * len(rows) + 1.5))
        ax.pcolormesh(tedges, np.arange(len(rows) + 1), np.ma.masked_invalid(np.array(rows)),
                      cmap=cmap, vmin=-0.5, vmax=2.5, shading="flat")
        for i, tk in enumerate(knees):
            if tk:
                ax.plot(tk, i + 0.5, "x", color=INK, ms=6)
        ax.set(xscale="log", yticks=np.arange(len(rows)) + 0.5, xlabel="t",
               title=f"J5: Adam normaliser regime (class majority), C = {C}, ε = 1e-8; × = knee")
        ax.set_yticklabels(labels, fontsize=7)
        ax.invert_yaxis()
        handles = [Line2D([], [], color=c, lw=6, label=l)
                   for c, l in zip(("#eb6834", "#86b6ef", "#104281"),
                                   ("ε floor", "background", "own events"))]
        ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.01, 1), frameon=False)
        fig.tight_layout()
        save(fig, f"J5_regime_map_C{C}")


def fig6():
    fig, axes = plt.subplots(1, len(K.KAPPAS), figsize=(3.2 * len(K.KAPPAS), 3.6), sharey=True)
    for ax, kappa in zip(axes, K.KAPPAS):
        for C, ls in ((128, "-"), (32, "--")):
            g = K.find("adam", C, kappa, 1e-8)
            f = K.FIELD / f"{g['name']}.npz"
            if not f.exists():
                continue
            z = np.load(f)
            for lane, b in enumerate(z["beta2s"]):
                ax.plot(z["time"], np.nanmean(z["ratio"][:, lane], -1), ls, color=BLUE[float(b)])
        ax.axhline(1, color=MUTED, lw=0.8)
        ax.set(xscale="log", title=f"κ = {kappa:g}", xlabel="t", ylim=(0, 1.3))
        polish_axis(ax)
    axes[0].set_ylabel("K2 balance ratio φ_cc m̄ / (r'²|χ_cc| + λ)")
    handles = [Line2D([], [], color=BLUE[b], label=f"β₂={b:g}") for b in K.BETA2S]
    handles += [Line2D([], [], color=INK, ls="-", label="C = 128"),
                Line2D([], [], color=INK, ls="--", label="C = 32")]
    axes[-1].legend(handles=handles, frameon=False, fontsize=7)
    fig.suptitle("J6: K2 balance (1 = stationary margins: a fixed point of the drift), ε = 1e-8")
    fig.tight_layout()
    save(fig, "J6_K2_balance")


def fig7():
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, C in zip(axes, (128, 32)):
        labs = [f"{b:g}" for b in K.BETA2S] + ["sgd"]
        w = 0.15
        for j, lab in enumerate(labs):
            vals = []
            for kappa in K.KAPPAS:
                g = K.find("sgd", C, kappa) if lab == "sgd" else K.find("adam", C, kappa, 1e-8)
                m = mk(g, lab) if g else {}
                vals.append(m.get("ell_T_rel", np.nan) / K.optimum(C, kappa)["ell_rel"])
            ax.bar(np.arange(len(K.KAPPAS)) + (j - 2) * w, vals, w,
                   color=SGD_C if lab == "sgd" else BLUE[float(lab)], label=lab)
        ax.axhline(1, color=MUTED, lw=0.8)
        ax.set(yscale="log", xticks=range(len(K.KAPPAS)),
               xticklabels=[f"κ={k:g}" for k in K.KAPPAS], title=f"C = {C}, ε = 1e-8, at T")
        polish_axis(ax)
    axes[0].set_ylabel("ℓ(T) / ℓ*")
    axes[1].legend(frameon=False)
    fig.suptitle("J7: late loss against the population optimum (κ ≤ 0.25: ℓ* is out of reach)")
    fig.tight_layout()
    save(fig, "J7_late_vs_optimum")


def fig8():
    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 4.6))
    for f in sorted(K.FIELD.glob("J_*_a0_*.npz")):
        z = np.load(f)
        d = K.load(f.stem)
        t_all = d["time"]
        for lane in range(z["total"].shape[1]):
            ell_state = z["loss_state"][:, lane].mean(-1)
            pred = z["total"][:, lane].mean(-1) / ell_state
            ell_mean = d["loss"][:, lane].mean((1, 2))
            k = t_all > 0
            meas = np.interp(z["time"], t_all[k], np.gradient(np.log(ell_mean[k]), t_all[k]))
            ok = (z["time"] >= 0.1) & (meas < 0) & (pred < 0)
            col = SGD_C if str(z["opt"]) == "sgd" else BLUE[float(z["beta2s"][lane])]
            a.plot(-meas[ok], -pred[ok], "o", color=col, ms=2.5, alpha=0.6)
    x = np.array([1e-5, 30])
    a.plot(x, x, color=MUTED, lw=1)
    a.fill_between(x, x * 0.75, x * 1.25, color="0.9", lw=0)
    a.set(xscale="log", yscale="log", xlabel="measured −d ln ℓ/dt (seed mean)",
          ylabel="frozen-field −d ln ℓ/dt at the state", title="(a) P1: field vs simulation (band ±25%)")
    names, parts = [], []
    for k_, p in NUM["P6"].items():
        if "C128" in k_ and ("_k1_" in k_ or "_k2_" in k_):
            names.append(k_.split("_a0_")[1].replace("_R32_s2026_f64", "").replace("_u4", "").replace("_u1", ""))
            parts.append([p["part_O"], p["part_B"], p["part_ito"]])
    parts = np.array(parts)
    if len(parts):
        y = np.arange(len(names))
        scale = np.abs(parts).sum(1)
        for j, (lab, col) in enumerate((("O part", "#104281"), ("B drift", "#86b6ef"), ("Itô", SGD_C))):
            b.barh(y + (j - 1) * 0.27, parts[:, j] / scale, 0.27, color=col, label=lab)
        b.set(yticks=y, xlabel="share of |O| + |B drift| + |Itô| (signed)",
              title="(b) P6: late d ℓ/dt split, C = 128, κ ∈ {1, 2}")
        b.set_yticklabels(names, fontsize=7)
        b.axvline(0, color=MUTED, lw=0.8)
        b.legend(frameon=False)
    for ax in (a, b):
        polish_axis(ax)
    fig.tight_layout()
    save(fig, "J8_field_P1_P6")


def main():
    only = set(sys.argv[1:])
    for name, fn in (("1", fig1), ("2", fig2), ("3", fig3), ("4", fig4), ("5", fig5),
                     ("6", fig6), ("7", fig7), ("8", fig8)):
        if not only or name in only:
            fn()


def _unused():
    fig5()
    fig6()
    fig7()
    fig8()


if __name__ == "__main__":
    main()
