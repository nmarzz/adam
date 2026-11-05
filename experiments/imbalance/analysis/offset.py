"""Why does the trajectory kink sit above the frozen one? (``imbalance.py offset``)

For every class ``c`` the half-loss time factorises as ``t_c = O_cc(t_c) / vbar_c`` with
``vbar_c = O_cc(t_c)/t_c`` the path-averaged drift of the own-class overlap. So

    log(t_A/t_S) = log(O_A/O_S)                                  [needed overlap]
                   - log(vbar_A/vbar_S) / (rate_A/rate_S)(Theta_0)  [drift decay along the path]
                   + log[(eta_S rate_S)/(eta_A rate_A)](Theta_0)    [frozen prediction]

where ``rate`` is the frozen ``|dO_cc/dt|/eta`` at ``Theta_0`` (``frozen/``). The frozen term
carries the predicted kink; the other two say whether Adam needs more overlap (a larger
``B``, i.e. noisier logits, at ``t_c``) or loses drift along the way. Also reported: the
logit-gap margin ``O_cc - mean_k O_kc`` and the logit noise ``B_cc`` at ``t_c``.
Writes ``offset.json`` and ``F_offset_*.png``.
"""
from __future__ import annotations

import json
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import lib
from lib import I

sys.path.insert(0, str(lib.REPO))
from experiments.plot_style import apply_paper_style, polish_axis  # noqa: E402

apply_paper_style()
plt.rcParams.update({"font.size": 12, "axes.titlesize": 12, "axes.labelsize": 12,
                     "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 9,
                     "lines.linewidth": 1.4})
BLUE = {0.9: "#6da7ec", 0.95: "#2a78d6", 0.99: "#1c5cab", 0.999: "#0d366b"}
MUTED = "#52514e"


def at_tc(time, series, tc):
    """``series`` (T, C) interpolated in log t at each class's own ``t_c``."""
    out = np.full(series.shape[1], np.nan)
    for c in range(series.shape[1]):
        if np.isfinite(tc[c]) and tc[c] > 0:
            out[c] = np.interp(np.log(tc[c]), np.log(time[1:]), series[1:, c])
    return out


def per_run(d, tc):
    t = d["time"]
    O, B = d["O"], d["B"]
    C = O.shape[-1]
    own = np.diagonal(O, axis1=1, axis2=2)                              # (T, C)
    other = (O.sum(1) - own) / (C - 1)                                  # mean_k!=c O_kc
    Bcc = np.diagonal(B, axis1=1, axis2=2)
    return {"O": at_tc(t, own, tc), "gap": at_tc(t, own - other, tc), "B": at_tc(t, Bcc, tc)}


def analyse(kappa, C, alpha, etaset="base"):
    s = lib.load_ode("m0", "sgd", kappa, C, alpha, etaset=etaset)
    fr = lib.load_frozen(kappa, C, alpha)
    if s is None or fr is None:
        return None
    pi = s["priors"]
    ts = lib.level_times(s["time"], s["loss"])
    S = per_run(s, ts)
    eta_s = I.eta_of("sgd", kappa, etaset)
    out = {}
    for b2 in I.BETA2S:
        a = lib.load_ode("m0", "adam", kappa, C, alpha, b2, etaset=etaset)
        if a is None or f"init_b2{b2:g}_adam" not in fr:
            continue
        eta_a = I.eta_of("adam", kappa, etaset)
        ta = lib.level_times(a["time"], a["loss"])
        A = per_run(a, ta)
        rate_a = np.abs(fr[f"init_b2{b2:g}_adam"])
        rate_s = np.abs(fr[f"init_b2{b2:g}_sgd"])
        frozen = np.log((eta_s * rate_s) / (eta_a * rate_a))          # log t_A/t_S if O equal
        total = np.log(ta / ts)
        need = np.log(A["O"] / S["O"])
        decay = total - need - frozen                                 # = -log(vbar ratio/frozen)
        out[b2] = {"pi": pi, "total": total, "frozen": frozen, "need": need, "decay": decay,
                   "gap_ratio": A["gap"] / S["gap"], "B_ratio": A["B"] / S["B"],
                   "O_A": A["O"], "O_S": S["O"], "kink_F": lib.kink(b2, C)}
    return out


def summarise(res, label):
    print(f"\n### {label}\n")
    print("| beta2 | band (u = pi/kink_F) | n | total log(t_A/t_S) | frozen | needed O | path decay |"
          " gap ratio | B_cc ratio |")
    print("|---|---|---|---|---|---|---|---|---|")
    rows = {}
    for b2, r in res.items():
        u = r["pi"] / r["kink_F"]
        ok = np.isfinite(r["total"]) & np.isfinite(r["need"])
        for name, m in (("head u > 3", u > 3), ("near 1/3-3", (u >= 1 / 3) & (u <= 3)),
                        ("tail u < 1/3", u < 1 / 3)):
            m = m & ok
            if not m.any():
                continue
            med = lambda x: float(np.median(x[m]))
            rows[f"{b2:g}_{name.split()[0]}"] = {k: med(r[k]) for k in
                                                 ("total", "frozen", "need", "decay")}
            print(f"| {b2:g} | {name} | {m.sum()} | {med(r['total']):+.3f} | {med(r['frozen']):+.3f} |"
                  f" {med(r['need']):+.3f} | {med(r['decay']):+.3f} | {med(r['gap_ratio']):.2f} |"
                  f" {med(r['B_ratio']):.2f} |")
    return rows


def figure(all_res):
    keys = [k for k in all_res if all_res[k]]
    if not keys:
        return
    fig, axes = plt.subplots(len(keys), 3, figsize=(14, 3.6 * len(keys)), squeeze=False)
    for row, key in zip(axes, keys):
        for b2, r in all_res[key].items():
            c = BLUE[b2]
            u = r["pi"] / r["kink_F"]
            row[0].plot(u, np.exp(r["total"]), "o", color=c, ms=3.5, label=f"β₂={b2:g}")
            row[0].plot(u, np.exp(r["frozen"]), "-", color=c, lw=1)
            row[1].plot(u, np.exp(r["need"]), "o", color=c, ms=3.5)
            row[2].plot(u, np.exp(r["decay"]), "o", color=c, ms=3.5)
        row[0].set(ylabel="t_A / t_S", title=f"{key}: measured (●) vs frozen at Θ₀ (—)")
        row[1].set(ylabel="O_cc(t_c): Adam / SGD", title="overlap needed at t_c")
        row[2].set(ylabel="frozen drift ratio / path-avg ratio",
                   title="drift lost along the path (Adam rel. to SGD)")
        for ax in row:
            ax.set(xscale="log", yscale="log", xlabel="u = π / kink_F")
            ax.axvline(1, color=MUTED, ls=":", lw=1)
            ax.axhline(1, color=MUTED, lw=0.6)
            polish_axis(ax)
        row[0].legend(frameon=False)
    fig.tight_layout()
    path = lib.OUTDIR / "F_offset"
    fig.savefig(path.with_suffix(".png"), dpi=130, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight", metadata={"CreationDate": None})
    plt.close(fig)
    print(f"\nsaved {path}.{{png,pdf}}")


def main():
    all_res, summary = {}, {}
    for kappa in I.KAPPAS:
        for C in I.CLASSES:
            for alpha in I.ALPHAS:
                res = analyse(kappa, C, alpha)
                if not res:
                    continue
                label = f"K = {kappa:g} I, C = {C}, alpha = {alpha:g}"
                summary[label] = summarise(res, label)
                if C >= 32:
                    all_res[f"κ={kappa:g}, C={C}, α={alpha:g}"] = res
    (lib.OUTDIR / "offset.json").write_text(json.dumps(lib.jsonable(summary), indent=1))
    figure(all_res)


if __name__ == "__main__":
    main()
