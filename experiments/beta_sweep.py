#!/usr/bin/env python3
"""Time-to-excess-risk heatmap from the repository's Adam ODE solver."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", "/tmp/adam-matplotlib")

import config
config.enable_x64()

import jax
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import dynamics
import problems
from experiments.diagonal_linreg import initialization, spectrum
from experiments.plot_style import apply_paper_style, polish_axis

apply_paper_style()


def first_hitting_time(risk, time, tolerance):
    risk, time = np.asarray(risk), np.asarray(time)
    reached = np.flatnonzero(risk <= tolerance)
    if not len(reached):
        return np.nan
    index = int(reached[0])
    if index == 0:
        return float(time[0])
    weight = (risk[index - 1] - tolerance) / max(risk[index - 1] - risk[index], 1e-14)
    return float(time[index - 1] + np.clip(weight, 0, 1) * (time[index] - time[index - 1]))


def run_sweep(args):
    problem = problems.get_problem("linreg")
    cov = spectrum(args.dimension, args.covariance, args.lambda_min,
                   args.lambda_max, args.spectral_decay)
    theta0, teacher = initialization(
        args.dimension, cov, args.target_norm, args.initial_excess_risk,
        args.init_overlap
    )
    beta1 = np.linspace(args.beta1_min, args.beta1_max, args.grid_size)
    beta2 = np.linspace(args.beta2_min, args.beta2_max, args.grid_size)
    hitting = np.full((args.grid_size, args.grid_size), np.nan)
    for row, b2 in enumerate(beta2):
        print(f"beta2 row {row + 1}/{args.grid_size}: {b2:.4f}")
        for column, b1 in enumerate(beta1):
            risk, time = dynamics.run_adam_ode(
                problem, theta0, teacher, cov, args.horizon, args.eta,
                beta1=float(b1), beta2=float(b2),
                dt=args.ode_step, num_samples=args.ode_samples,
                history_length=args.ode_history,
                diffusion_samples=args.diffusion_samples,
                diffusion_history=args.diffusion_history,
                eps=args.epsilon, noise_std=args.noise_std,
                key=jax.random.PRNGKey(args.seed),
            )
            risk.block_until_ready()
            hitting[row, column] = first_hitting_time(
                risk, time, args.risk_tolerance
            )
    return beta1, beta2, hitting, np.asarray(cov)


def plot(beta1, beta2, hitting, args, output):
    display = np.where(np.isfinite(hitting), hitting, args.horizon)
    figure, axis = plt.subplots(figsize=(9.0, 7.2), constrained_layout=True)
    image = axis.imshow(
        display, origin="lower", aspect="auto", interpolation="nearest",
        extent=[beta1[0], beta1[-1], beta2[0], beta2[-1]],
        cmap="viridis_r", vmin=0, vmax=args.horizon,
    )
    missing = (~np.isfinite(hitting)).astype(float)
    if np.any(missing):
        axis.contourf(beta1, beta2, missing, levels=[0.5, 1.5],
                      colors="none", hatches=["////"])
    colorbar = figure.colorbar(image, ax=axis, pad=0.025, fraction=0.055)
    colorbar.set_label(r"First time $P(t)\leq\varepsilon_{\rm risk}$")
    colorbar.ax.tick_params(labelsize=15, width=1.15, length=5)
    axis.set_xlabel(r"$\beta_1$")
    axis.set_ylabel(r"$\beta_2$")
    axis.set_title(
        rf"ODE time to excess-risk tolerance $\varepsilon_{{\rm risk}}={args.risk_tolerance:g}$"
    )
    if np.any(missing):
        axis.text(0.99, 0.02, f"hatched: not reached by T={args.horizon:g}",
                  transform=axis.transAxes, ha="right", va="bottom", fontsize=13,
                  color="white",
                  bbox={"facecolor": "black", "alpha": 0.45,
                        "edgecolor": "none", "pad": 3})
    polish_axis(axis, grid=False)
    figure.savefig(output.with_suffix(".png"), dpi=300, bbox_inches="tight")
    figure.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dimension", type=int, default=512)
    parser.add_argument("--covariance", choices=["isotropic", "decaying"], default="decaying")
    parser.add_argument("--grid-size", type=int, default=11)
    parser.add_argument("--beta1-min", type=float, default=0.0)
    parser.add_argument("--beta1-max", type=float, default=0.9)
    parser.add_argument("--beta2-min", type=float, default=0.0)
    parser.add_argument("--beta2-max", type=float, default=0.95)
    parser.add_argument("--risk-tolerance", type=float, default=0.25)
    parser.add_argument("--horizon", type=float, default=8.0)
    parser.add_argument("--eta", type=float, default=0.35)
    parser.add_argument("--epsilon", type=float, default=0.1)
    parser.add_argument("--noise-std", type=float, default=0.5)
    parser.add_argument("--target-norm", type=float, default=1.0)
    parser.add_argument("--initial-excess-risk", type=float, default=0.6)
    parser.add_argument("--init-overlap", type=float, default=0.2)
    parser.add_argument("--lambda-min", type=float, default=0.25)
    parser.add_argument("--lambda-max", type=float, default=2.0)
    parser.add_argument("--spectral-decay", type=float, default=4.0)
    parser.add_argument("--ode-step", type=float, default=0.05)
    parser.add_argument("--ode-samples", type=int, default=3_000)
    parser.add_argument("--ode-history", type=int, default=100)
    parser.add_argument("--diffusion-samples", type=int, default=750)
    parser.add_argument("--diffusion-history", type=int, default=35)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "figs" / "experiments")
    return parser.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    beta1, beta2, hitting, cov = run_sweep(args)
    stem = args.output_dir / f"time_to_epsilon_{args.covariance}_D{args.dimension}"
    plot(beta1, beta2, hitting, args, stem)
    np.savez_compressed(stem.with_suffix(".npz"), beta1=beta1, beta2=beta2,
                        hitting_time=hitting, spectrum=cov)
    with (args.output_dir / "beta_sweep_config.json").open("w") as handle:
        json.dump({key: str(value) if isinstance(value, Path) else value
                   for key, value in vars(args).items()}, handle, indent=2)


if __name__ == "__main__":
    main()
