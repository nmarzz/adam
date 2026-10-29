#!/usr/bin/env python3
"""Check the teacher terms of the GMM ODEs: linear regression with mixture inputs.

y = theta*^T x + omega with x | c ~ N(xbar_c, I), two non-symmetric means,
class weights (0.3, 0.7), a teacher that overlaps the means, omega ~ N(0, 0.5^2).
Here f(q; omega) = q_1 - q_2 - omega ignores the label, so the drift through
``chi`` only comes from the class-dependent mean prediction error, and the
B_12 equation receives the forcing ``chi^T Xbar^T theta*``.  With K = I the
system closes in (B_11, B_12, O_1):

    dB11/dt = -eta Sym([B11|B12] phi + O1 chi) + eta^2 sigma
    dB12/dt = -eta (phi^T [B12; B22] + chi^T Xbar^T theta*)
    dO1/dt  = -eta (phi^T O + chi^T Xbar^T Xbar)

Usage: python experiments/gmm_teacher_check.py --dimensions 500 1000 2000 4000
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import config
config.enable_x64()

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

import gmm

BETA1, BETA2, EPS, LR, NOISE, HORIZON, DT = 0.9, 0.95, 1e-2, 0.5, 0.5, 5.0, 0.02
PROBS = jnp.array([0.3, 0.7])


def build(D, seed=5):
    rng = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(rng.standard_normal((D, 4)))
    means = np.stack([1.2 * Q[:, 0], 0.6 * Q[:, 0] + 0.8 * Q[:, 1]], axis=1)
    teacher = 1.0 * Q[:, 1] + 0.5 * Q[:, 2]
    theta0 = 0.4 * Q[:, 3]
    return jnp.asarray(means), jnp.asarray(teacher), jnp.asarray(theta0)


def risk(B11, B12, B22, O1, O2):
    """0.5 E[(q1 - q2 - omega)^2] with q1 - q2 | c ~ N(O1_c - O2_c, B11 - 2B12 + B22)."""
    return 0.5 * (jnp.sum(PROBS * (O1 - O2) ** 2) + B11 - 2 * B12 + B22 + NOISE ** 2)


def simulate(D, runs, means, teacher, theta0, key):
    record = D // 10

    def one(key):
        def step(carry, k):
            th, m, v, key = carry
            key, k1, k2, k3 = jax.random.split(key, 4)
            c = jax.random.categorical(k1, jnp.log(PROBS))
            x = means[:, c] + jax.random.normal(k2, (D,))
            y = teacher @ x + NOISE * jax.random.normal(k3)
            g = x * (th @ x - y)
            m = BETA1 * m + (1 - BETA1) * g
            v = BETA2 * v + (1 - BETA2) * g ** 2
            return (th - (LR / D) * m / jnp.sqrt(v + EPS), m, v, key), None

        def chunk(carry, start):
            th = carry[0]
            obs = risk(th @ th, th @ teacher, teacher @ teacher, th @ means, teacher @ means)
            carry, _ = lax.scan(step, carry, start + jnp.arange(record))
            return carry, obs

        z = jnp.zeros(D)
        _, out = lax.scan(chunk, (theta0, z, z, key), jnp.arange(int(HORIZON * 10)) * record)
        return out

    return np.asarray(jax.jit(jax.vmap(one))(jax.random.split(key, runs)))


def make_coefficients(key, n=256, burn=400, used=2048, tail=200):
    L = burn + used + tail
    ku, kz, kx, kw = jax.random.split(key, 4)
    U, Z = jax.random.uniform(ku, (L, n)), jax.random.normal(kz, (L, n, 2))
    XI, W = jax.random.normal(kx, (L, n)), jax.random.normal(kw, (L, n))
    cb = (1 - BETA1) / np.sqrt(1 - BETA2)

    @jax.jit
    def coefficients(B, O):
        c = jnp.minimum(jnp.searchsorted(jnp.cumsum(PROBS), U), len(PROBS) - 1)
        q = O.T[c] + Z @ gmm.psd_sqrt(B).T
        f = q[..., 0] - q[..., 1] - NOISE * W
        _, S = lax.scan(lambda s, h: (BETA2 * s + h,) * 2, jnp.zeros(n), f ** 2 * XI ** 2)
        S = S + EPS / (1 - BETA2)

        def backward(carry, x):
            R1 = x[0] + BETA1 * carry[0]
            R3 = x[1] + BETA1 * BETA2 * carry[1]
            return (R1, R3), (R1, R3)

        _, (R1, R3) = lax.scan(backward, (jnp.zeros(n), jnp.zeros(n)),
                               (S ** -0.5, S ** -1.5), reverse=True)
        sl = slice(burn, burn + used)
        f, xi, R1, R3 = f[sl], XI[sl], R1[sl], R3[sl]
        weight = cb * xi ** 2 * (R1 - f ** 2 * xi ** 2 * R3)
        phi = jnp.array([[1.0], [-1.0]]) * jnp.mean(weight)            # grad_q f = (1, -1)
        onehot = jax.nn.one_hot(c[sl], len(PROBS))
        chi = cb * jnp.einsum("tnc,tn->c", onehot, f * xi ** 2 * R1)[:, None] / (used * n)
        sigma = jnp.mean((cb * f * xi * R1) ** 2)
        return phi, chi, sigma

    return coefficients


def integrate_ode(means, teacher, theta0, coefficients):
    gram, cross = means.T @ means, (means.T @ teacher)[:, None]
    B22, O2 = teacher @ teacher, (teacher @ means)[None, :]

    def rhs(state):
        B11, B12, O1 = state
        B = jnp.array([[B11[0, 0], B12[0, 0]], [B12[0, 0], B22]])
        O = jnp.concatenate([O1, O2])
        phi, chi, sigma = coefficients(B, O)
        X = jnp.concatenate([B11, B12], axis=1) @ phi + O1 @ chi
        return (-LR * (X + X.T) + LR ** 2 * sigma,
                -LR * (phi.T @ jnp.array([[B12[0, 0]], [B22]]) + chi.T @ cross),
                -LR * (phi.T @ O + chi.T @ gram))

    state = ((theta0 @ theta0)[None, None], (theta0 @ teacher)[None, None],
             (theta0 @ means)[None, :])
    out = []
    for _ in range(int(round(HORIZON / DT)) + 1):
        B11, B12, O1 = state
        out.append(float(risk(B11[0, 0], B12[0, 0], B22, O1[0], O2[0])))
        k1 = rhs(state)
        k2 = rhs(tuple(a + DT * b for a, b in zip(state, k1)))
        state = tuple(a + 0.5 * DT * (b + c) for a, b, c in zip(state, k1, k2))
    return np.arange(len(out)) * DT, np.array(out)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dimensions", type=int, nargs="+", default=[500, 1000, 2000, 4000])
    parser.add_argument("--runs", type=int, default=64)
    args = parser.parse_args()
    coefficients = make_coefficients(jax.random.PRNGKey(9))
    for D in args.dimensions:
        means, teacher, theta0 = build(D)
        risks = simulate(D, args.runs, means, teacher, theta0, jax.random.PRNGKey(0))
        t_ode, ode = integrate_ode(means, teacher, theta0, coefficients)
        t = np.arange(risks.shape[1]) / 10
        gap = risks.mean(0) - np.interp(t, t_ode, ode)
        se = risks.std(0, ddof=1) / np.sqrt(args.runs)
        k = np.argmax(np.abs(gap))
        print(f"D={D}: sup_t |mean Adam - ODE| = {abs(gap[k]):.4f} at t={t[k]:.1f} "
              f"(s.e. there {se[k]:.4f}); risk {ode[0]:.3f} -> {ode[-1]:.3f}", flush=True)


if __name__ == "__main__":
    main()
