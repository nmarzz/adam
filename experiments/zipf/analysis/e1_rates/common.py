"""Shared helpers for the E1a frozen-rate analysis (numpy only, reads rates.npz).

predicted_rates / kappa_beta are copied verbatim (formulas) from
experiments/zipf_scaling.py so that jax is not imported."""
import os

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
DATA = REPO + "/figs/experiments/zipf/rates.npz"
OUTDIR = REPO + "/figs/experiments/zipf/analysis/e1_rates"
os.makedirs(OUTDIR, exist_ok=True)
MEAN_NORM = 5.0
BETA1 = 0.9

CONFIGS = [(0.95, 1, 1e-8), (0.99, 1, 1e-8), (0.999, 1, 1e-8),
           (0.99, 4, 1e-8), (0.99, 16, 1e-8), (0.99, 64, 1e-8),
           (0.99, 1, 1e-4), (0.99, 1, 1e-2), (0.99, 16, 1e-4), (0.99, 16, 1e-2)]


def tag(b2, b, e):
    return f"b2{b2:g}_b{b}_e{e:g}"


def kappa_beta(beta1, beta2):
    return (np.pi / 2) * ((1 - beta1 / np.sqrt(beta2)) / (1 - beta1)) ** 2


def predicted_rates(priors, beta1, beta2, eps, batch, r=MEAN_NORM):
    pi = np.asarray(priors)
    C = len(pi)
    mean_f = 1 - 1 / C
    second = pi * mean_f ** 2 + (1 - pi) / C ** 2
    c_beta = (1 - beta1) / np.sqrt(1 - beta2)
    return {
        "sgd": r ** 2 * pi * mean_f,
        "psgd": r ** 2 * pi * mean_f / np.sqrt(second / batch + eps),
        "rare": (r ** 2 * batch * pi * c_beta * np.sqrt(2 / np.pi)
                 / (1 - beta1 / np.sqrt(beta2))),
        "kink_beta2": kappa_beta(beta1, beta2) * (1 - beta2) / batch,
        "kink_eps": batch * eps,
        "kink_background": 1 / C ** 2,
    }


def load():
    return dict(np.load(DATA))
