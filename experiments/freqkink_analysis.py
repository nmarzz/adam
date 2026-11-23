"""Analysis of the frequency-kink runs (experiments/freqkink.py).

Per-token quantities are pooled into log-frequency bins of the empirical
training frequency pi_c. Two curves per bin and checkpoint:

* loss      l_bin(t)  pooled held-out cross-entropy of the bin's tokens,
* gain      G_bin(t)  pooled  z_c - z_u on the token's own held-out contexts, with u a
                      token never seen in training. Every column without an own event
                      equals the common background column, so G isolates the effect of
                      the token's own events and l_c = (lse - z_u) - G_c exactly.
* signal    S_bin(t)  pooled  w_c . (x_c - xbar) (contaminated by the background push).

Rate estimates per bin (all per optimizer step):

* ``Gslope`` stationary slope dG/dt on [T/4, T]; for the frozen-drift runs this is
            the per-class drift at initialization (use with --frozen),
* ``G``     early slope of the own-event gain, G_bin(t_w) / t_w (default),
* ``Gt``    theta / t_theta with t_theta the first step at which G_bin >= theta,
* ``S``     early slope of the signal S_bin(t_w) / t_w,
* ``half``  1 / t_half with t_half the first step at which l_bin <= l_bin(0) / 2,
* ``drop``  delta / t_delta with t_delta the first step at which l_bin <= l_bin(0) - delta.

Figures:

1. rate versus pi_c for every batch size, one panel per optimizer setting, with
   the predicted kink pi* = kappa (1 - beta2) / b and reference slopes 1/2 and 1;
2. self-normalized ratio versus x = b pi_c / (1 - beta2) (collapse test), with a
   fitted two-segment bend and the predicted x = kappa;
3. per-output sqrt(v_hat) against pi_c with the eps levels.

Usage (from the repository root):
    real/bin/python experiments/freqkink_analysis.py --rate S --tw 100
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from plot_style import apply_paper_style, polish_axis

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(os.environ.get("FREQKINK_OUT", ROOT / "results" / "freqkink"))
FIGS = Path(os.environ.get("FREQKINK_FIGS", ROOT / "figs" / "freqkink"))
C = 50257


def kappa(beta1=0.0, beta2=0.99):
    return np.pi / 2 * ((1 - beta1 / np.sqrt(beta2)) / (1 - beta1)) ** 2


# --------------------------------------------------------------------------- #
# Loading and binning
# --------------------------------------------------------------------------- #
def load_runs(pattern="*"):
    runs = []
    for f in sorted(glob.glob(str(OUT / "runs" / f"{pattern}.npz"))):
        d = np.load(f)
        r = {k: d[k] for k in d.files if k != "cfg"}
        r["cfg"] = json.loads(str(d["cfg"]))
        r["name"] = Path(f).stem
        runs.append(r)
    return runs


def bins_for(train_counts, width=0.25, min_count=1):
    pi = train_counts / train_counts.sum()
    u = np.log10(np.where(train_counts >= min_count, pi, np.nan))
    lo = np.floor(np.nanmin(u) / width) * width
    hi = np.ceil(np.nanmax(u) / width) * width
    edges = np.arange(lo, hi + width / 2, width)
    k = np.digitize(u, edges)
    k[~np.isfinite(u)] = -1
    return pi, edges, k


def bin_curves(r, edges, k, min_test=50):
    """Pooled loss and signal per bin, shapes (n_bins, n_ckpt); plus bin centres."""
    tc = r["test_counts"]
    L, S, G, cen, mass, ntypes = [], [], [], [], [], []
    pi = r["train_counts"] / r["train_counts"].sum()
    for j in range(1, len(edges)):
        m = (k == j) & (tc > 0)
        n = tc[m].sum()
        if n < min_test:
            continue
        L.append(r["per_tok_sum"][:, m].sum(1) / n)
        S.append((r["own_logit_sum"][:, m].sum(1) - (r["mean_ctx_logit"][:, m] * tc[m]).sum(1)) / n)
        if "gain_sum" in r:
            G.append(r["gain_sum"][:, m].sum(1) / n)
        elif "bg_logit_sum" in r:
            G.append((r["own_logit_sum"][:, m].sum(1) - r["bg_logit_sum"][:, m].sum(1)) / n)
        # centre: test-occurrence-weighted mean of log10 pi over the bin
        cen.append(10 ** (np.sum(tc[m] * np.log10(pi[m])) / n))
        mass.append(pi[k == j].sum())
        ntypes.append(int(m.sum()))
    return dict(L=np.array(L), S=np.array(S), G=np.array(G), pi=np.array(cen), mass=np.array(mass),
                ntypes=np.array(ntypes))


def first_crossing(steps, y, level):
    """First step (log-interpolated) at which y <= level; nan if never."""
    below = np.nonzero(y <= level)[0]
    below = below[below > 0]
    if len(below) == 0:
        return np.nan
    i = below[0]
    s0, s1 = max(steps[i - 1], 0.5), steps[i]
    y0, y1 = y[i - 1], y[i]
    f = 0.0 if y0 == y1 else (y0 - level) / (y0 - y1)
    return float(np.exp(np.log(s0) + f * (np.log(s1) - np.log(s0))))


def interp_at(steps, y, t):
    return float(np.interp(np.log(t), np.log(np.maximum(steps, 0.5)), y))


def first_rise(steps, y, level):
    return first_crossing(steps, -y, -level)


def rates(r, bc, kind="G", tw=100, delta=1.0, theta=1.0):
    steps = r["steps"].astype(float)
    out = []
    for j in range(len(bc["pi"])):
        if kind in ("S", "G"):
            out.append(interp_at(steps, bc[kind][j], tw) / tw)
        elif kind == "Gslope":
            # stationary drift: least-squares slope of G against steps on [T/4, T]
            sel = steps >= steps[-1] / 4
            out.append(np.polyfit(steps[sel], bc["G"][j][sel], 1)[0])
        elif kind == "Gt":
            out.append(theta / first_rise(steps, bc["G"][j], theta))
        elif kind == "half":
            l0 = bc["L"][j, 0]
            out.append(1.0 / first_crossing(steps, bc["L"][j], l0 / 2))
        elif kind == "drop":
            l0 = bc["L"][j, 0]
            out.append(delta / first_crossing(steps, bc["L"][j], l0 - delta))
        else:
            raise ValueError(kind)
    return np.array(out)


# --------------------------------------------------------------------------- #
# Fits
# --------------------------------------------------------------------------- #
def fit_bend(x, y, grid=None):
    """Continuous two-segment fit of log10 y on log10 x.

    log y = a + s_hi * max(u - u_k, 0) + s_lo * min(u - u_k, 0).
    Returns dict(u_k, a, s_lo, s_hi, sse).
    """
    ok = np.isfinite(x) & np.isfinite(y) & (y > 0) & (x > 0)
    u, v = np.log10(x[ok]), np.log10(y[ok])
    if len(u) < 5:
        return None
    grid = np.linspace(np.quantile(u, 0.1), np.quantile(u, 0.9), 200) if grid is None else grid
    best = None
    for uk in grid:
        A = np.stack([np.ones_like(u), np.maximum(u - uk, 0), np.minimum(u - uk, 0)], 1)
        coef, *_ = np.linalg.lstsq(A, v, rcond=None)
        sse = float(np.sum((A @ coef - v) ** 2))
        if best is None or sse < best["sse"]:
            best = dict(u_k=float(uk), a=float(coef[0]), s_hi=float(coef[1]), s_lo=float(coef[2]),
                        sse=sse, n=int(len(u)))
    return best


def bootstrap_bend(x, y, n_boot=300, seed=0):
    rng = np.random.default_rng(seed)
    ok = np.nonzero(np.isfinite(x) & np.isfinite(y) & (y > 0))[0]
    uks = []
    for _ in range(n_boot):
        i = rng.choice(ok, len(ok))
        f = fit_bend(x[i], y[i])
        if f is not None:
            uks.append(f["u_k"])
    return np.quantile(uks, [0.05, 0.95]) if uks else (np.nan, np.nan)


def slope(x, y, lo=None, hi=None):
    ok = np.isfinite(x) & np.isfinite(y) & (y > 0)
    if lo is not None:
        ok &= x >= lo
    if hi is not None:
        ok &= x <= hi
    if ok.sum() < 3:
        return np.nan
    return float(np.polyfit(np.log10(x[ok]), np.log10(y[ok]), 1)[0])


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def group_key(cfg):
    if cfg["opt"] == "sgd":
        return "SGD"
    return rf"Adam $\beta_2$={cfg['beta2']}, $\epsilon$={cfg['eps']:.0e}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rate", default="G", choices=["G", "Gslope", "Gt", "S", "half", "drop"])
    ap.add_argument("--frozen", action="store_true", help="analyse the frozen-drift (lr -> 0) runs")
    ap.add_argument("--theta", type=float, default=1.0, help="gain threshold for --rate Gt")
    ap.add_argument("--tw", type=float, default=100, help="early window (steps) for --rate S")
    ap.add_argument("--delta", type=float, default=1.0)
    ap.add_argument("--width", type=float, default=0.25, help="bin width in decades")
    ap.add_argument("--center", action="store_true", help="analyse the centred-feature runs")
    ap.add_argument("--b-ref", type=int, default=None, help="reference batch (default: largest)")
    ap.add_argument("--tag", default="")
    ap.add_argument("--norm", default="window", choices=["window", "plateau"],
                    help="collapse normalization: sqrt(b) window law vs reference, or each run's plateau")
    args = ap.parse_args()
    apply_paper_style()
    FIGS.mkdir(parents=True, exist_ok=True)

    runs = [r for r in load_runs() if bool(r["cfg"].get("center", False)) == args.center
            and bool(r["cfg"].get("frozen", False)) == args.frozen]
    if not runs:
        raise SystemExit("no runs")
    pi, edges, k = bins_for(runs[0]["train_counts"], args.width)
    groups = {}
    for r in runs:
        r["bc"] = bin_curves(r, edges, k)
        r["rate"] = rates(r, r["bc"], args.rate, args.tw, args.delta, args.theta)
        groups.setdefault(group_key(r["cfg"]), []).append(r)
    for g in groups.values():
        g.sort(key=lambda r: r["cfg"]["b"])
    bs = sorted({r["cfg"]["b"] for r in runs})
    cmap = plt.get_cmap("viridis")
    col = {b: cmap(i / max(len(bs) - 1, 1) * 0.9) for i, b in enumerate(bs)}
    suffix = f"_{args.rate}" + (f"_tw{args.tw:g}" if args.rate in ("S", "G") else "") + \
        (f"_th{args.theta:g}" if args.rate == "Gt" else "") + ("_frozen" if args.frozen else "") + ("_plateau" if args.norm == "plateau" else "") + \
        ("_center" if args.center else "") + args.tag
    report = {"rate": args.rate, "tw": args.tw, "theta": args.theta, "center": args.center,
              "frozen": args.frozen, "groups": {}}

    # ---- Figure 1: rate vs frequency -------------------------------------
    names = sorted(groups, key=lambda s: (s == "SGD", s))
    fig, axes = plt.subplots(1, len(names), figsize=(6.2 * len(names), 5.4), squeeze=False)
    for ax, gname in zip(axes[0], names):
        rep = report["groups"].setdefault(gname, {"runs": {}})
        for r in groups[gname]:
            b, lr = r["cfg"]["b"], r["cfg"]["lr"]
            x, y = r["bc"]["pi"], r["rate"] / lr
            ok = np.isfinite(y) & (y > 0)
            ax.loglog(x[ok], y[ok], "o-", ms=4, lw=1.6, color=col[b], label=f"b={b}")
            neg = np.isfinite(y) & (y <= 0)
            if neg.any():
                ax.plot(x[neg], np.full(neg.sum(), np.nanmin(y[ok]) if ok.any() else 1), "x",
                        color=col[b], ms=5, alpha=0.6)
            info = {"lr": lr, "steps": int(r["steps"][-1]), "final_loss": float(r["total"][-1])}
            if r["cfg"]["opt"] == "adam":
                ps = kappa(0.0, r["cfg"]["beta2"]) * (1 - r["cfg"]["beta2"]) / b
                ax.axvline(ps, color=col[b], ls="--", lw=1.2, alpha=0.8)
                info.update(pi_star_pred=ps, slope_above=slope(x, y, lo=ps), slope_below=slope(x, y, hi=ps))
                f = fit_bend(x, y)
                if f:
                    info["bend_fit"] = f
            else:
                info["slope_all"] = slope(x, y)
            rep["runs"][f"b={b}"] = info
        # reference slopes anchored at the top-right of the data
        allx = np.concatenate([r["bc"]["pi"] for r in groups[gname]])
        ally = np.concatenate([r["rate"] / r["cfg"]["lr"] for r in groups[gname]])
        okk = np.isfinite(ally) & (ally > 0)
        if okk.any():
            x0, y0 = np.nanmax(allx), np.nanmax(ally[okk])
            xs = np.array([x0 / 1e4, x0])
            ax.loglog(xs, y0 * 3 * (xs / x0) ** 0.5, color="0.4", ls=":", lw=1.4, label="slope 1/2")
            ax.loglog(xs, y0 * 3 * (xs / x0) ** 1.0, color="0.15", ls="-.", lw=1.4, label="slope 1")
        ax.set_xlabel(r"token frequency $\pi_c$")
        ax.set_title(gname, fontsize=15)
        polish_axis(ax)
    lab = {"G": rf"own-event gain rate $G(t_w)/(\eta t_w)$, $t_w$={args.tw:g}",
           "Gslope": r"init drift $\dot G/\eta$ (frozen, per step)",
           "Gt": rf"$\theta/(\eta\, t_\theta)$, $G(t_\theta)=\theta$={args.theta:g}",
           "S": rf"early signal rate $S(t_w)/(\eta t_w)$, $t_w$={args.tw:g}",
           "half": r"$1/(\eta\, t_{1/2})$", "drop": rf"$\delta/(\eta\, t_\delta)$, $\delta$={args.delta:g}"}
    axes[0, 0].set_ylabel(lab[args.rate])
    axes[0, 0].legend(fontsize=11, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIGS / f"rate_vs_freq{suffix}.pdf", metadata={"CreationDate": None})
    fig.savefig(FIGS / f"rate_vs_freq{suffix}.png")
    plt.close(fig)

    # ---- Figure 2: collapse ----------------------------------------------
    adam = [g for g in names if g != "SGD" and len(groups[g]) > 1]
    if adam:
        fig, axes = plt.subplots(1, len(adam), figsize=(6.2 * len(adam), 5.4), squeeze=False)
        for ax, gname in zip(axes[0], adam):
            g = groups[gname]
            b_ref = args.b_ref or max(r["cfg"]["b"] for r in g)
            ref = [r for r in g if r["cfg"]["b"] == b_ref][0]
            beta2 = ref["cfg"]["beta2"]
            X, Y = [], []
            for r in g:
                if r is ref and args.norm == "window":
                    continue
                b = r["cfg"]["b"]
                if args.norm == "window":
                    # window law: rate per step ∝ eta sqrt(b pi)  =>  normalize so the window is flat
                    R = (r["rate"] / r["cfg"]["lr"]) / (ref["rate"] / ref["cfg"]["lr"]) * np.sqrt(b_ref / b)
                else:
                    # each run divided by its own plateau (median rate over pi_c in [1e-4, 1e-2])
                    pl = (r["bc"]["pi"] >= 1e-4) & (r["bc"]["pi"] <= 1e-2)
                    R = r["rate"] / np.nanmedian(r["rate"][pl])
                x = b * r["bc"]["pi"] / (1 - beta2)
                ok = np.isfinite(R) & (R > 0)
                ax.loglog(x[ok], R[ok], "o", ms=5, color=col[b],
                          label=f"b={b} / {b_ref}" if args.norm == "window" else f"b={b}")
                X.append(x[ok]); Y.append(R[ok])
            if not X:
                continue
            X, Y = np.concatenate(X), np.concatenate(Y)
            f = fit_bend(X, Y)
            kap = kappa(0.0, beta2)
            ax.axvline(kap, color="k", ls="--", lw=1.4, label=r"predicted $x=\kappa_\beta$")
            if f:
                lo, hi = bootstrap_bend(X, Y)
                xs = np.logspace(np.log10(X.min()), np.log10(X.max()), 200)
                us = np.log10(xs)
                ax.loglog(xs, 10 ** (f["a"] + f["s_hi"] * np.maximum(us - f["u_k"], 0)
                                     + f["s_lo"] * np.minimum(us - f["u_k"], 0)),
                          color="C3", lw=2, label=rf"fit: bend $x$={10 ** f['u_k']:.2g}")
                ax.axvspan(10 ** lo, 10 ** hi, color="C3", alpha=0.12)
                report["groups"][gname]["collapse"] = dict(f, x_k=10 ** f["u_k"], x_k_90ci=[10 ** lo, 10 ** hi],
                                                           kappa=kap, b_ref=b_ref)
            ax.set_xlabel(r"$x=b\pi_c/(1-\beta_2)$")
            ax.set_title(gname, fontsize=15)
            polish_axis(ax)
        axes[0, 0].set_ylabel(r"$\frac{\mathrm{rate}(b)/\eta_b}{\mathrm{rate}(b_\mathrm{ref})/\eta_\mathrm{ref}}\sqrt{b_\mathrm{ref}/b}$"
                              if args.norm == "window" else "rate / plateau rate")
        axes[0, 0].legend(fontsize=11)
        fig.tight_layout()
        fig.savefig(FIGS / f"collapse{suffix}.pdf", metadata={"CreationDate": None})
        fig.savefig(FIGS / f"collapse{suffix}.png")
        plt.close(fig)

    # ---- Figure 3: sqrt(v) vs eps ----------------------------------------
    vr = [r for r in runs if "v_med" in r]
    if vr:
        fig, axes = plt.subplots(1, len(vr), figsize=(4.6 * len(vr), 4.6), squeeze=False, sharey=True)
        for ax, r in zip(axes[0], vr):
            st = r["steps"][1:]
            for frac, c in ((0.1, "C0"), (0.5, "C1"), (1.0, "C3")):
                i = min(int(frac * (len(st) - 1)), len(st) - 1)
                ok = r["train_counts"] > 0
                ax.loglog(pi[ok], r["v_med"][i][ok], ".", ms=1.5, color=c, alpha=0.35,
                          label=f"step {st[i]:,}", rasterized=True)
            ax.axhline(r["cfg"]["eps"], color="k", ls="--", lw=1.2)
            ax.set_title(f"b={r['cfg']['b']}, eps={r['cfg']['eps']:.0e}", fontsize=12)
            ax.set_xlabel(r"$\pi_c$")
            polish_axis(ax)
        axes[0, 0].set_ylabel(r"median$_j\sqrt{\hat v_{jc}}$")
        axes[0, 0].legend(fontsize=9, markerscale=6)
        fig.tight_layout()
        fig.savefig(FIGS / f"sqrtv_vs_eps{'_center' if args.center else ''}.png")
        plt.close(fig)

    (OUT / f"analysis{suffix}.json").write_text(json.dumps(report, indent=1, default=float))
    print(json.dumps(report, indent=1, default=lambda v: round(float(v), 4)))


if __name__ == "__main__":
    main()
