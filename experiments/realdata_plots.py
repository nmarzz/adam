"""Grid figure of Adam-on-real-data vs the single-Gaussian ODE.

    real/bin/python experiments/realdata_plots.py grid --names A B ... --betas 0.9,0.999 0.9,0.95 \
        --eta-rel 0.1 --out realdata_grid.png
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results" / "realdata"
FIGS = ROOT / "figs" / "experiments" / "realdata"

LABELS = {
    "cifar_animal_sketch_2048": "CIFAR-10, sphere + Gaussian sketch",
    "cifar_animal_tanh_2048": "CIFAR-10, tanh RF",
    "cifar_animal_relu+sketch_2048": "CIFAR-10, ReLU RF + sketch",
    "owt_logfreq_raw_768": "GPT-2 / OWT, raw hidden state",
    "owt_logfreq_sketchs_768": "GPT-2 / OWT, sphere + sketch",
    "owt_logfreq_sketchzs_768": "GPT-2 / OWT, standardize + sphere + sketch",
    "msd_year_relu+sketch_1024": "MSD year, ReLU RF + sketch",
    "msd_year_relu+sketchs_1024": "MSD year, sphere + ReLU RF + sketch",
}


def find(name, b1, b2, eta_rel, extra=""):
    for path in sorted(glob.glob(str(RES / f"run_{name}_eta*_b1{b1:g}_b2{b2:g}_*{extra}.npz"))):
        z = np.load(path)
        a = json.loads(str(z["args"]))
        if abs(a.get("eta_rel", 0) - eta_rel) < 1e-9 and a["name"] == name:
            return z, a
    return None, None


def rel_err(t, P, to, Po, tmin):
    sel = t >= tmin
    Pi = np.exp(np.interp(t[sel], to, np.log(Po)))
    r = np.abs(P.mean(1)[sel] - Pi) / Pi
    return float(np.median(r)), float(r.max())


def grid(args):
    names, betas = args.names, [tuple(map(float, b.split(","))) for b in args.betas]
    fig, axes = plt.subplots(len(names), len(betas), figsize=(4.2 * len(betas), 3.1 * len(names)),
                             squeeze=False)
    for i, name in enumerate(names):
        for j, (b1, b2) in enumerate(betas):
            ax = axes[i, j]
            z, a = find(name, b1, b2, args.eta_rel)
            if z is None:
                ax.set_axis_off(); continue
            for mode, c in (("real", "C0"), ("twin_emp", "C2")):
                if f"{mode}_t" not in z:
                    continue
                t, P = z[f"{mode}_t"][1:], z[f"{mode}_P"][1:]
                lo, hi = np.percentile(P, [10, 90], axis=1)
                ax.fill_between(t, lo, hi, color=c, alpha=0.2, lw=0)
                ax.plot(t, P.mean(1), color=c, lw=1.4,
                        label="Adam, real data" if mode == "real" else "Adam, Gaussian twin")
            to, Po = z["ode_emp_t"][1:], z["ode_emp_P"][1:]
            ax.plot(to, Po, "k--", lw=1.6, label="ODE")
            if "ode_gauss_P" in z:
                ax.plot(z["ode_gauss_t"][1:], z["ode_gauss_P"][1:], color="C3", ls=":", lw=1.4,
                        label="ODE, Gaussian noise")
            md, mx = rel_err(z["real_t"], z["real_P"], z["ode_emp_t"], z["ode_emp_P"], 0.05)
            ax.text(0.03, 0.04, f"median err {100 * md:.1f}%, max {100 * mx:.0f}%",
                    transform=ax.transAxes, fontsize=8)
            ax.set_xscale("log"); ax.set_yscale("log")
            if i == 0:
                ax.set_title(rf"$\beta_1$={b1:g}, $\beta_2$={b2:g}", fontsize=11)
            if j == 0:
                ax.set_ylabel(LABELS.get(name, name) + "\nexcess risk", fontsize=8)
            if i == len(names) - 1:
                ax.set_xlabel("t = steps / D")
            if i == 0 and j == 0:
                ax.legend(fontsize=7, loc="upper right")
    fig.tight_layout()
    FIGS.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGS / args.out, dpi=130)
    print("wrote", FIGS / args.out)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("grid")
    g.add_argument("--names", nargs="+", required=True)
    g.add_argument("--betas", nargs="+", default=["0.9,0.999", "0.9,0.95", "0.0,0.9"])
    g.add_argument("--eta-rel", type=float, default=0.1)
    g.add_argument("--out", default="realdata_grid.png")
    args = ap.parse_args()
    grid(args)


if __name__ == "__main__":
    main()
