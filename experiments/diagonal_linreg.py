#!/usr/bin/env python3
"""Noisy linear regression under isotropic or decaying diagonal covariance.

All optimizer and ODE calculations are delegated to the repository's
``simulate``, ``dynamics``, and ``problems`` modules. This file only constructs
the experiment, extracts risk/B11/B12, and plots central 80% run envelopes.
"""
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
import jax.numpy as jnp
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import dynamics
import problems
import simulate
from utils import compute_ci


def spectrum(d: int, kind: str, lower: float, upper: float, decay: float):
    if kind == "isotropic":
        return jnp.ones(d)
    u = jnp.linspace(0.0, 1.0, d)
    raw = (jnp.exp(-decay * u) - jnp.exp(-decay)) / (1.0 - jnp.exp(-decay))
    return lower + (upper - lower) * raw


def initialization(d: int, cov, target_norm: float, initial_excess_risk: float,
                   overlap: float):
    """Fix the initial risk and B geometry in whitened coordinates for every D."""
    if initial_excess_risk <= 0:
        raise ValueError("initial_excess_risk must be positive")
    if not -1.0 < overlap < 1.0:
        raise ValueError("init_overlap must lie strictly between -1 and 1")
    teacher_direction = jnp.ones(d) / jnp.sqrt(d)
    teacher_direction /= jnp.linalg.norm(teacher_direction)
    orthogonal = jnp.cos(2 * jnp.pi * (jnp.arange(d) + 0.5) / d)
    orthogonal -= teacher_direction * (teacher_direction @ orthogonal)
    orthogonal /= jnp.linalg.norm(orthogonal)
    teacher_white = target_norm * teacher_direction
    radicand = (
        2.0 * initial_excess_risk
        - target_norm**2 * (1.0 - overlap**2)
    )
    if radicand < 0:
        minimum = 0.5 * target_norm**2 * (1.0 - overlap**2)
        raise ValueError(
            f"initial_excess_risk must be at least {minimum:g} for this "
            "target norm and overlap"
        )
    init_norm = target_norm * overlap + np.sqrt(radicand)
    if init_norm <= 0:
        raise ValueError("the requested geometry gives a nonpositive initial norm")
    initial_white = init_norm * (
        overlap * teacher_direction + jnp.sqrt(1.0 - overlap**2) * orthogonal
    )
    # Remove the last floating-point dependence on D: enforce the requested
    # excess risk directly through ||theta_0-theta_*||_Sigma^2 / 2.
    error_white = initial_white - teacher_white
    error_white *= jnp.sqrt(
        initial_excess_risk / (0.5 * jnp.sum(error_white**2))
    )
    initial_white = teacher_white + error_white
    return (initial_white[:, None] / jnp.sqrt(cov[:, None]),
            teacher_white[:, None] / jnp.sqrt(cov[:, None]))


def subsample(path, source_time, display_time):
    indices = np.searchsorted(np.asarray(source_time), display_time, side="left")
    indices = np.clip(indices, 0, len(source_time) - 1)
    return np.asarray(path)[..., indices]


def run_case(d: int, kind: str, args, problem):
    cov = spectrum(d, kind, args.lambda_min, args.lambda_max, args.spectral_decay)
    theta0, teacher = initialization(
        d, cov, args.target_norm, args.initial_excess_risk, args.init_overlap
    )
    steps = int(round(args.horizon * d))
    sim = simulate.build_adam(
        problem, cov, teacher, args.eta / d,
        beta1=args.beta1, beta2=args.beta2,
        eps=args.epsilon * cov[:, None],
        label_noise_std=args.noise_std,
    )
    keys = jax.random.split(jax.random.PRNGKey(args.seed + d), args.runs)
    adam_risk, adam_B = simulate.run_many_with_B(
        sim, problem, theta0, teacher, cov, steps, keys=keys
    )
    adam_risk.block_until_ready()
    adam_iteration = np.arange(steps)

    ode_risk, ode_B, ode_time = dynamics.run_adam_ode(
        problem, theta0, teacher, cov, args.horizon, args.eta,
        beta1=args.beta1, beta2=args.beta2,
        dt=args.ode_step, num_samples=args.ode_samples,
        history_length=args.ode_history,
        diffusion_samples=args.diffusion_samples,
        diffusion_history=args.diffusion_history,
        eps=args.epsilon, noise_std=args.noise_std, return_B=True,
        key=jax.random.PRNGKey(args.seed + 10_000 + d),
    )
    ode_risk.block_until_ready()
    ode_iteration = np.asarray(ode_time) * d
    last_iteration = int(np.floor(min(adam_iteration[-1], ode_iteration[-1])))
    display_iteration = np.unique(
        np.rint(np.linspace(0, last_iteration, args.evaluations)).astype(int)
    )
    noise_floor = 0.5 * args.noise_std**2
    return {
        "iteration": display_iteration,
        "adam_risk": subsample(adam_risk, adam_iteration, display_iteration) + noise_floor,
        "adam_B11": subsample(adam_B[..., 0, 0], adam_iteration, display_iteration),
        "adam_B12": subsample(adam_B[..., 0, 1], adam_iteration, display_iteration),
        "ode_risk": np.interp(display_iteration, ode_iteration, np.asarray(ode_risk))
                    + noise_floor,
        "ode_B11": np.interp(display_iteration, ode_iteration,
                              np.asarray(ode_B[..., 0, 0])),
        "ode_B12": np.interp(display_iteration, ode_iteration,
                              np.asarray(ode_B[..., 0, 1])),
        "spectrum": np.asarray(cov),
    }


def plot(kind: str, rows, output: Path):
    metrics = [("risk", "Population risk"), ("B11", r"$B_{11}$"), ("B12", r"$B_{12}$")]
    figure, axes = plt.subplots(1, 3, figsize=(12.2, 3.7), squeeze=False)
    colors = plt.cm.viridis(np.linspace(0.12, 0.82, len(rows)))
    for color, (d, result) in zip(colors, rows):
        for column, (key, title) in enumerate(metrics):
            _, lower, upper = compute_ci(jnp.asarray(result[f"adam_{key}"]), alpha=0.2)
            axis = axes[0, column]
            axis.fill_between(result["iteration"], lower, upper,
                              color=color, alpha=0.20, linewidth=0)
            axis.plot(result["iteration"], result[f"ode_{key}"],
                      color=color, lw=2, label=f"ODE, D = {d}")
            axis.grid(alpha=0.18)
            axis.set_title(title)
            axis.set_xlabel(r"Algorithm iteration $k$")
    axes[0, 0].fill_between([], [], [], color="0.5", alpha=0.20,
                            label="Adam central 80%")
    axes[0, 0].legend(frameon=False)
    label = ("Isotropic covariance" if kind == "isotropic"
             else "Lower-bounded decaying diagonal covariance")
    figure.suptitle(label, y=0.995)
    figure.tight_layout()
    figure.savefig(output.with_suffix(".png"), dpi=220, bbox_inches="tight")
    figure.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dimensions", type=int, nargs="+", default=[256, 512, 1024])
    parser.add_argument("--covariances", choices=["isotropic", "decaying"], nargs="+",
                        default=["isotropic", "decaying"])
    parser.add_argument("--runs", type=int, default=80)
    parser.add_argument("--evaluations", type=int, default=151)
    parser.add_argument("--horizon", type=float, default=3.0)
    parser.add_argument("--eta", type=float, default=0.35)
    parser.add_argument("--beta1", type=float, default=0.8)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--epsilon", type=float, default=0.1)
    parser.add_argument("--noise-std", type=float, default=0.5)
    parser.add_argument("--target-norm", type=float, default=1.0)
    parser.add_argument("--initial-excess-risk", type=float, default=0.6)
    parser.add_argument("--init-overlap", type=float, default=0.2)
    parser.add_argument("--lambda-min", type=float, default=0.25)
    parser.add_argument("--lambda-max", type=float, default=2.0)
    parser.add_argument("--spectral-decay", type=float, default=4.0)
    parser.add_argument("--ode-step", type=float, default=0.04)
    parser.add_argument("--ode-samples", type=int, default=4_000)
    parser.add_argument("--ode-history", type=int, default=120)
    parser.add_argument("--diffusion-samples", type=int, default=1_000)
    parser.add_argument("--diffusion-history", type=int, default=40)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "figs" / "experiments")
    return parser.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    problem = problems.get_problem("linreg")
    for kind in args.covariances:
        rows = []
        for d in args.dimensions:
            print(f"{kind}: D={d}, runs={args.runs}")
            result = run_case(d, kind, args, problem)
            rows.append((d, result))
            np.savez_compressed(args.output_dir / f"linreg_{kind}_D{d}.npz", **result)
        plot(kind, rows, args.output_dir / f"linreg_{kind}_dimension_envelopes")
    with (args.output_dir / "linreg_config.json").open("w") as handle:
        json.dump({key: str(value) if isinstance(value, Path) else value
                   for key, value in vars(args).items()}, handle, indent=2)


if __name__ == "__main__":
    main()
