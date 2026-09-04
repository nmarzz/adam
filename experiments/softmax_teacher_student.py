#!/usr/bin/env python3
"""Multiclass softmax teacher--student experiment using repository APIs.

The class count is a parameter and defaults to C=2,3,5. Optimizer simulation,
the matrix-valued ODE, risk evaluation, and B construction all live in the
shared repository modules; this file is only experiment setup and plotting.
"""
from __future__ import annotations

import argparse
import json
import math
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
from experiments.diagonal_linreg import spectrum, subsample
from experiments.plot_style import apply_paper_style, polish_axis
from utils import compute_ci

apply_paper_style()


def initialization(d, classes, cov, target_scale, init_scale, overlap, seed):
    """Reference-class logits with the same initial B geometry across D."""
    rank = classes - 1
    if d < 2 * rank:
        raise ValueError("dimension must be at least 2*(classes-1)")
    key_teacher, key_other = jax.random.split(jax.random.PRNGKey(seed + classes))
    teacher_basis, _ = jnp.linalg.qr(jax.random.normal(key_teacher, (d, rank)))
    other = jax.random.normal(key_other, (d, rank))
    other -= teacher_basis @ (teacher_basis.T @ other)
    other_basis, _ = jnp.linalg.qr(other)
    teacher_white = target_scale * teacher_basis
    student_basis = overlap * teacher_basis + math.sqrt(1.0 - overlap**2) * other_basis
    student_white = init_scale * student_basis
    return (student_white / jnp.sqrt(cov[:, None]),
            teacher_white / jnp.sqrt(cov[:, None]))


def summarize_B(B, classes):
    rank = classes - 1
    return {
        "B11_trace": np.trace(np.asarray(B)[..., :rank, :rank], axis1=-2, axis2=-1) / rank,
        "B12_trace": np.trace(np.asarray(B)[..., :rank, rank:], axis1=-2, axis2=-1) / rank,
    }


def run_case(d, classes, args, problem):
    cov = spectrum(d, args.covariance, args.lambda_min,
                   args.lambda_max, args.spectral_decay)
    theta0, teacher = initialization(
        d, classes, cov, args.target_scale, args.init_scale,
        args.init_overlap, args.seed
    )
    steps = int(round(args.horizon * d))
    sim = simulate.build_adam(
        problem, cov, teacher, args.eta / d,
        beta1=args.beta1, beta2=args.beta2,
        eps=args.epsilon * cov[:, None],
    )
    keys = jax.random.split(
        jax.random.PRNGKey(args.seed + 100 * classes + d), args.runs
    )
    adam_risk, adam_B = simulate.run_many_with_B(
        sim, problem, theta0, teacher, cov, steps, keys=keys
    )
    adam_risk.block_until_ready()
    adam_time = np.arange(steps) / d

    ode_risk, ode_B, ode_time = dynamics.run_adam_ode(
        problem, theta0, teacher, cov, args.horizon, args.eta,
        beta1=args.beta1, beta2=args.beta2,
        dt=args.ode_step, num_samples=args.ode_samples,
        history_length=args.ode_history,
        diffusion_samples=args.diffusion_samples,
        diffusion_history=args.diffusion_history,
        eps=args.epsilon, return_B=True,
        key=jax.random.PRNGKey(args.seed + 10_000 + classes + d),
    )
    ode_risk.block_until_ready()
    display_time = np.linspace(
        0.0, min(adam_time[-1], float(ode_time[-1])), args.evaluations
    )
    adam_summary = summarize_B(adam_B, classes)
    ode_summary = summarize_B(ode_B, classes)
    return {
        "time": display_time,
        "adam_risk": subsample(adam_risk, adam_time, display_time),
        "adam_B11_trace": subsample(adam_summary["B11_trace"], adam_time, display_time),
        "adam_B12_trace": subsample(adam_summary["B12_trace"], adam_time, display_time),
        "ode_risk": subsample(ode_risk, ode_time, display_time),
        "ode_B11_trace": subsample(ode_summary["B11_trace"], ode_time, display_time),
        "ode_B12_trace": subsample(ode_summary["B12_trace"], ode_time, display_time),
        "adam_B": np.moveaxis(
            subsample(np.moveaxis(np.asarray(adam_B), 1, -1), adam_time, display_time),
            -1, 1,
        ),
        "ode_B": np.moveaxis(
            subsample(np.moveaxis(np.asarray(ode_B), 0, -1), ode_time, display_time),
            -1, 0,
        ),
        "spectrum": np.asarray(cov),
    }


def plot(classes, rows, output):
    metrics = [
        ("risk", "Population cross-entropy"),
        ("B11_trace", r"$\frac{\mathrm{tr}(B_{11})}{C-1}$"),
        ("B12_trace", r"$\frac{\mathrm{tr}(B_{12})}{C-1}$"),
    ]
    figure, axes = plt.subplots(
        len(rows), 3, figsize=(16.0, 4.25 * len(rows)),
        sharex=True, squeeze=False, constrained_layout=True
    )
    for row, (d, result) in enumerate(rows):
        for column, (key, title) in enumerate(metrics):
            _, lower, upper = compute_ci(jnp.asarray(result[f"adam_{key}"]), alpha=0.2)
            axis = axes[row, column]
            axis.fill_between(result["time"], lower, upper,
                              color="#E45756", alpha=0.32, linewidth=0)
            axis.plot(result["time"], result[f"ode_{key}"],
                      color="#111111", lw=3.0)
            polish_axis(axis)
            if row == 0:
                axis.set_title(title)
            if column == 0:
                axis.set_ylabel(f"$D={d}$")
            if row == len(rows) - 1:
                axis.set_xlabel(r"Rescaled time $t=k/D$")
    axes[0, 0].plot([], [], color="#111111", lw=3.0, label="ODE")
    axes[0, 0].fill_between([], [], [], color="#E45756", alpha=0.32,
                            label="Adam central 80%")
    axes[0, 0].legend(frameon=False, loc="best")
    figure.suptitle(f"{classes}-class softmax teacher--student")
    figure.savefig(output.with_suffix(".png"), dpi=300, bbox_inches="tight")
    figure.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--classes", type=int, nargs="+", default=[2, 3, 5])
    parser.add_argument("--dimensions", type=int, nargs="+", default=[256, 512])
    parser.add_argument("--runs", type=int, default=40)
    parser.add_argument("--evaluations", type=int, default=81)
    parser.add_argument("--covariance", choices=["isotropic", "decaying"], default="isotropic")
    parser.add_argument("--horizon", type=float, default=2.0)
    parser.add_argument("--eta", type=float, default=0.3)
    parser.add_argument("--beta1", type=float, default=0.8)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--epsilon", type=float, default=0.1)
    parser.add_argument("--target-scale", type=float, default=1.2)
    parser.add_argument("--init-scale", type=float, default=0.7)
    parser.add_argument("--init-overlap", type=float, default=0.15)
    parser.add_argument("--lambda-min", type=float, default=0.25)
    parser.add_argument("--lambda-max", type=float, default=2.0)
    parser.add_argument("--spectral-decay", type=float, default=4.0)
    parser.add_argument("--ode-step", type=float, default=0.05)
    parser.add_argument("--ode-samples", type=int, default=3_000)
    parser.add_argument("--ode-history", type=int, default=100)
    parser.add_argument("--diffusion-samples", type=int, default=600)
    parser.add_argument("--diffusion-history", type=int, default=30)
    parser.add_argument("--risk-samples", type=int, default=4_000)
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "figs" / "experiments")
    return parser.parse_args()


def main():
    args = parse_args()
    if any(classes < 2 for classes in args.classes):
        raise ValueError("class counts must be at least two")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for classes in args.classes:
        problem = problems.make_softmax_problem(
            classes, risk_samples=args.risk_samples,
            risk_seed=args.seed + classes
        )
        rows = []
        for d in args.dimensions:
            print(f"classes={classes}, D={d}, runs={args.runs}")
            result = run_case(d, classes, args, problem)
            rows.append((d, result))
            np.savez_compressed(args.output_dir / f"softmax_C{classes}_D{d}.npz", **result)
        plot(classes, rows, args.output_dir / f"softmax_C{classes}_{args.covariance}_envelopes")
    with (args.output_dir / "softmax_config.json").open("w") as handle:
        json.dump({key: str(value) if isinstance(value, Path) else value
                   for key, value in vars(args).items()}, handle, indent=2)


if __name__ == "__main__":
    main()
