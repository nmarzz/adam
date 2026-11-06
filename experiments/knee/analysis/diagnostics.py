"""Series J diagnostics of a softmax-mixture state ``(B, O)`` (no bias, ``K = Id`` primed units).

Conventions (``experiments/imbalance.py``, ``paper/zipf_adam.tex``):

* ``O[:, c]`` is the mean logit vector of class ``c`` and ``B`` the logit covariance, so
  ``q | c ~ N(O[:, c], B)``. Both are invariant under the ``kappa`` rescaling
  (``x' = x / sqrt(kappa)``, ``Theta' = sqrt(kappa) Theta``): the cached series I ODE files
  store them unchanged, and the same arrays are the primed-unit order parameters.
* Primed units: ``K = Id``, means ``r' q_c`` with ``r' = 5 / sqrt(kappa)``,
  ``eta'_Adam = sqrt(kappa) eta``, ``eta'_SGD = kappa eta``; Adam's ``eps`` is K-scaled and
  invariant (``1e-8``).
* Signal / noise: ``O = r' Theta_par^T Q`` so ``P = O O^T / r'^2`` and ``N = B - P``.

Contents
--------
``margins``, ``diff_var``, ``noise``, ``geometry``           (closed form, numpy, batched)
``class_moments``                                              (Gaussian MC, jax, common random numbers)
``normaliser``                                                 (own / bg / R / tau / z / regime labels)
``balance``                                                    (E4 weighted margin and balance ratio)
``ito_split``                                                  (d ell_c / dt = O part + B drift part + Ito part)
``shape_loss`` / ``ShapeTable``                                (exact L_C(m, s^2) by a Frullani integral)
"""
from __future__ import annotations

import functools
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
for p in (str(REPO), str(REPO / "experiments"), str(REPO / "experiments" / "imbalance" / "analysis")):
    if p not in sys.path:
        sys.path.insert(0, p)
os.environ.setdefault("MPLCONFIGDIR", "/tmp/adam-matplotlib")
# the GPU is shared: no up-front allocation of 75% of its memory
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
from jax import lax  # noqa: E402

R = 5.0
BETA1 = 0.9
EPS = 1e-8
N_MC = 2 ** 15
SEED = 20261106


def r_primed(kappa):
    return R / np.sqrt(kappa)


def eta_primed(opt, kappa, eta):
    return eta * (np.sqrt(kappa) if opt == "adam" else kappa)


# --------------------------------------------------------------------------- #
# Closed-form geometry (numpy, any leading batch dims)
# --------------------------------------------------------------------------- #
def margins(O):
    """``m_c = O_cc - mean_{k != c} O_kc`` and ``m_min_c = O_cc - max_{k != c} O_kc``."""
    O = np.asarray(O, dtype=float)
    C = O.shape[-1]
    d = np.diagonal(O, axis1=-2, axis2=-1)
    m = d - (O.sum(-2) - d) / (C - 1)
    off = np.where(np.eye(C, dtype=bool), -np.inf, O)
    return m, d - off.max(-2)


def diff_var(B):
    """``s_c^2 = mean_{k != c} Var(q_c - q_k) = mean_{k != c}(B_cc + B_kk - 2 B_ck)``.
    (The ``k = c`` term is 0; averaging over all ``k`` would only scale by ``(C-1)/C``.)"""
    B = np.asarray(B, dtype=float)
    C = B.shape[-1]
    d = np.diagonal(B, axis1=-2, axis2=-1)
    V = d[..., :, None] + d[..., None, :] - 2 * B
    return V.sum(-1) / (C - 1)


def noise(B, O, r):
    """``N = B - O O^T / r^2`` and its differential part ``P N P`` (``P = I - 11^T/C``)."""
    B, O = np.asarray(B, dtype=float), np.asarray(O, dtype=float)
    C = B.shape[-1]
    N = B - O @ np.swapaxes(O, -1, -2) / r ** 2
    P = np.eye(C) - 1.0 / C
    return N, P @ N @ P


def geometry(B, O, r):
    """Margins, logit-difference variances and ``gamma_c = (s_c^2 - s_noise,c^2) r^2 / m_c^2``.

    ``s_c^2 - s_noise,c^2 = mean_{k != c} |O_c. - O_k.|^2 / r^2`` is the signal part of the
    logit-difference variance (rows of ``O``)."""
    m, m_min = margins(O)
    s2 = diff_var(B)
    N, Nd = noise(B, O, r)
    s2n = diff_var(N)
    with np.errstate(divide="ignore", invalid="ignore"):
        gamma = (s2 - s2n) * r ** 2 / m ** 2
    return {"m": m, "m_min": m_min, "s2": s2, "s2_noise": s2n, "gamma": gamma,
            "noise_trace": np.trace(Nd, axis1=-2, axis2=-1)}


# --------------------------------------------------------------------------- #
# Gaussian Monte Carlo with fixed common random numbers
# --------------------------------------------------------------------------- #
@functools.lru_cache(maxsize=None)
def draws(C, n=N_MC, seed=SEED):
    """Fixed standard normals ``(n, C)`` (antithetic pairs: the odd moments of the noise
    cancel exactly)."""
    g = np.random.default_rng(seed + 1009 * C).standard_normal((n // 2, C))
    return jnp.asarray(np.concatenate([g, -g]))


def _sym_sqrt(B):
    vals, vecs = jnp.linalg.eigh((B + B.T) / 2)
    return (vecs * jnp.sqrt(jnp.clip(vals, 0.0))) @ vecs.T


@functools.partial(jax.jit, static_argnames=("hessian",))
def _class_moments(B, O, Z, hessian=False):
    noise_ = Z @ _sym_sqrt(B)
    C = B.shape[0]
    n = Z.shape[0]

    def one(c):
        q = O[:, c][None, :] + noise_                           # (n, C)
        lse = jax.nn.logsumexp(q, axis=1)
        p = jnp.exp(q - lse[:, None])
        own = jax.nn.one_hot(c, C, dtype=bool)
        a = jax.nn.logsumexp(jnp.where(own[None, :], -jnp.inf, q), axis=1) - q[:, c]
        loss = jnp.logaddexp(0.0, a)                            # -log p_c, no cancellation
        g = jax.nn.sigmoid(a)                                   # 1 - p_c, no cancellation
        err = (jnp.argmax(q, axis=1) != c).astype(q.dtype)
        out = {"loss": loss.mean(), "loss_sd": loss.std(), "g1": g.mean(), "g1_sd": g.std(),
               "g2": (g ** 2).mean(), "err": err.mean(), "p1": p.mean(0),
               "p2": (p ** 2).mean(0)}
        if hessian:
            out["H"] = jnp.diag(p.mean(0)) - p.T @ p / n
        return out

    return lax.map(one, jnp.arange(C))


def class_moments(B, O, *, n=N_MC, seed=SEED, hessian=False):
    """Per-class MC moments at one state: ``loss`` (``ell_c``), ``g1 = E[g|c]``,
    ``g2 = E[g^2|c]`` (``g = 1 - p_c``, computed as a sigmoid of the competitor
    log-sum-exp, accurate down to ``g ~ 1e-300``), ``err``, ``p1[k, :] = E[p|k]``,
    ``p2[k, c] = E[p_c^2 | k]`` and, optionally, ``H[c] = E[diag(p) - p p^T | c]``.
    Standard errors: ``*_se`` (antithetic pairs make them conservative)."""
    B, O = jnp.asarray(B, dtype=jnp.float64), jnp.asarray(O, dtype=jnp.float64)
    out = {k: np.asarray(v) for k, v in _class_moments(B, O, draws(B.shape[0], n, seed),
                                                        hessian=hessian).items()}
    out["loss_se"] = out.pop("loss_sd") / np.sqrt(n)
    out["g1_se"] = out.pop("g1_sd") / np.sqrt(n)
    return out


# --------------------------------------------------------------------------- #
# Normaliser decomposition and regime labels
# --------------------------------------------------------------------------- #
@functools.lru_cache(maxsize=None)
def _g_table():
    import lib  # experiments/imbalance/analysis/lib.py
    la = np.linspace(np.log(1e-14), np.log(1e14), 561)
    return la, np.log(lib.g(np.exp(la)))


def psi_sat(beta2, beta1=BETA1):
    """Saturated (sign-like) kick per own event: ``c_beta sqrt(2/pi)/(1 - beta1/sqrt(beta2))``
    (``lib.coef_beta``)."""
    c_beta = (1 - beta1) / np.sqrt(1 - beta2)
    return c_beta * np.sqrt(2 / np.pi) / (1 - beta1 / np.sqrt(beta2))


def F_state(z, beta2, beta1=BETA1, tol=1e-8):
    """``lib.F_pred`` with ``a / f^2`` replaced by ``1 / z^2``: the background/eps prefactor
    of the saturated rate for an own event of size ``g`` against a background second
    moment ``m_bg`` (``z = g sqrt(1-beta2)/sqrt(m_bg + eps)``). At ``Theta_0`` this is
    exactly ``lib.F_pred(beta2, C)``."""
    rho = beta1 / np.sqrt(beta2)
    lags = int(np.ceil(np.log(tol) / np.log(rho)))
    l = np.arange(lags)
    la, lg = _g_table()
    z = np.atleast_1d(np.asarray(z, dtype=float))
    a = -2 * np.log(np.maximum(z, 1e-300))[:, None] - l[None, :] * np.log(beta2)
    gl = np.exp(np.interp(np.clip(a, la[0], la[-1]), la, lg))
    return (gl * rho ** l).sum(1) / (np.sqrt(2 / np.pi) * np.sum(rho ** l))


def normaliser(mom, priors, *, beta2, eps=EPS, kink=None):
    """Normaliser decomposition of Adam's per-output second moment (``b = 1``, ``K = Id``):

    * ``own_c = pi_c E[(1 - p_c)^2 | c]``, ``bg_c = sum_{k != c} pi_k E[p_c^2 | k]``,
      ``R_c = sqrt(own_c + bg_c + eps)``;
    * ``tau_c = E[g|c] / sqrt(E[g^2|c])``;
    * ``z_c = E[g|c] sqrt(1 - beta2) / sqrt(bg_c/(1 - pi_c) + eps)`` (``E_{k != c}`` is the
      conditional background mean; at ``Theta_0`` this reproduces ``lib.F_pred``);
    * per-``pi_c r'^2`` forcing rates: ``window_c = E[g|c] / R_c`` (deterministic normaliser)
      and ``sat_c = psi_sat F(z_c)`` (rare-event law); ``state_regime`` = the smaller one
      (they cross exactly at ``lib.kink`` at ``Theta_0`` with ``bg << own``);
    * ``norm_regime``: ``eps`` if ``eps > own + bg``, else ``background`` if ``bg > own``,
      else ``own``; ``kink_regime``: ``window`` if ``pi_c >= kink`` (``lib.kink(beta2, C)``).
    """
    priors = np.asarray(priors, dtype=float)
    C = len(priors)
    p2 = mom["p2"]
    own = priors * mom["g2"]
    bg = priors @ p2 - priors * np.diag(p2)
    Rn = np.sqrt(own + bg + eps)
    tau = mom["g1"] / np.sqrt(mom["g2"])
    z = mom["g1"] * np.sqrt(1 - beta2) / np.sqrt(bg / (1 - priors) + eps)
    window = mom["g1"] / Rn
    sat = psi_sat(beta2) * F_state(z, beta2)
    if kink is None:
        import lib
        kink = lib.kink(beta2, C)
    norm_regime = np.where(eps > own + bg, "eps", np.where(bg > own, "background", "own"))
    return {"own": own, "bg": bg, "R": Rn, "tau": tau, "z": z, "rate_window": window,
            "rate_sat": sat, "state_regime": np.where(sat < window, "saturated", "window"),
            "kink_regime": np.where(priors >= kink, "window", "saturated"),
            "norm_regime": norm_regime, "kink": kink}


def diagnose(B, O, priors, *, r, beta2, eps=EPS, n=N_MC, seed=SEED):
    """All per-class diagnostics of one state (geometry + MC moments + normaliser)."""
    out = geometry(B, O, r)
    mom = class_moments(B, O, n=n, seed=seed)
    out.update({k: mom[k] for k in ("loss", "loss_se", "g1", "g1_se", "g2", "err")})
    out.update(normaliser(mom, priors, beta2=beta2, eps=eps))
    return out


# --------------------------------------------------------------------------- #
# E4 balance and Ito split (need the full coefficient matrices)
# --------------------------------------------------------------------------- #
def balance(O, phi, chi, r2):
    """E4 (``paper/zipf_adam.tex``, eq. zipf_fp). With ``D = phi^T O + r^2 chi^T``
    (``dO/dt = -eta D``) and ``1^T phi = 0``:

    * ``mbar_c = sum_{s != c} (-phi_sc / phi_cc)(O_cc - O_sc)`` so ``D_cc = phi_cc mbar_c + r^2 chi_cc``;
    * ``lambda_c = mean_{s != c} D_sc`` (common-mode drift of column ``c``);
    * ``ratio_c = phi_cc mbar_c / (r^2 |chi_cc| + lambda_c)``: 1 iff the margin
      ``O_cc - mean_{s != c} O_sc`` is stationary; ``ratio_plain`` uses the plain margin;
    * ``mbar_pred = (r^2 |chi_cc| + lambda_c) / phi_cc``.
    ``margin_rate = -(D_cc - lambda_c)`` (per ``eta``) equals ``(forcing + lambda)(1 - ratio)``.
    """
    O, phi, chi = (np.asarray(a, dtype=float) for a in (O, phi, chi))
    C = O.shape[0]
    D = phi.T @ O + r2 * chi.T
    phicc = np.diag(phi)
    off = ~np.eye(C, dtype=bool)
    gaps = np.diag(O)[None, :] - O
    mbar = np.where(off, -phi * gaps, 0.0).sum(0) / phicc
    lam = np.where(off, D, 0.0).sum(0) / (C - 1)
    forcing = r2 * np.abs(np.diag(chi))
    m, _ = margins(O)
    return {"mbar": mbar, "lambda": lam, "forcing": forcing, "phi_cc": phicc,
            "chi_cc": np.diag(chi), "mbar_pred": (forcing + lam) / phicc,
            "ratio": phicc * mbar / (forcing + lam), "ratio_plain": phicc * m / (forcing + lam),
            "margin_rate": -(np.diag(D) - lam), "col_sum_phi": np.abs(phi.sum(0)).max()}


def ito_split(B, O, phi, chi, sigma, *, r2, eta, mom=None, n=N_MC, seed=SEED):
    """``d ell_c/dt`` split into the ``O`` part ``E[f|c] . dO[:, c]``, the ``B`` drift part
    ``1/2 <E[H|c], -eta Sym(B phi + O chi)>`` and the Ito part ``1/2 <E[H|c], eta^2 sigma>``
    (heat-equation identity ``d E h / dB = 1/2 E[Hess h]``; ``H = diag(p) - p p^T``)."""
    B, O, phi, chi, sigma = (np.asarray(a, dtype=float) for a in (B, O, phi, chi, sigma))
    if mom is None or "H" not in mom:
        mom = class_moments(B, O, n=n, seed=seed, hessian=True)
    C = B.shape[0]
    dO = -eta * (phi.T @ O + r2 * chi.T)
    A = B @ phi + O @ chi
    dB_drift = -eta * (A + A.T)
    dB_ito = eta ** 2 * sigma
    f_mean = mom["p1"] - np.eye(C)                     # row c: E[p - e_c | c]
    part_O = np.einsum("ck,kc->c", f_mean, dO)
    part_B = 0.5 * np.einsum("cij,ij->c", mom["H"], dB_drift)
    part_ito = 0.5 * np.einsum("cij,ij->c", mom["H"], dB_ito)
    return {"O": part_O, "B_drift": part_B, "ito": part_ito,
            "total": part_O + part_B + part_ito}


# --------------------------------------------------------------------------- #
# Exact shape function L_C(m, s^2) of the symmetric state
# --------------------------------------------------------------------------- #
def _logsumexp(a, axis):
    mx = np.max(a, axis=axis, keepdims=True)
    mx = np.where(np.isfinite(mx), mx, 0.0)
    return np.squeeze(mx, axis) + np.log(np.sum(np.exp(a - mx), axis=axis))


def _lambda_table(C, sig, u):
    """``Lambda(u) = E log(1 + e^{-u} sum_{k<C} e^{sig z_k})`` on ``u`` by
    ``log(1+S) = int_0^inf (1 - e^{-tS}) e^{-t} dt / t``:
    ``Lambda(u) = int (1 - (1 - h(t e^{-u}))^{C-1}) e^{-t} dln t``, ``h(S) = E[1 - exp(-S e^{sig z})]``."""
    dz = min(0.02, 0.1 / max(sig, 1e-9))
    z = np.arange(-13.0, 13.0 + sig + dz / 2, dz)
    logw = -z ** 2 / 2 - 0.5 * np.log(2 * np.pi) + np.log(dz)
    log_mgf = _logsumexp(sig * z + logw, 0)                  # log E e^{sig z} (on the grid)
    lo, hi = -sig * (13 + sig) - 25.0, np.log(40.0) + 9.5 * sig + 8.0
    lnS = np.linspace(lo, hi, 3001)
    logh = np.empty_like(lnS)
    log1mh = np.empty_like(lnS)
    for i0 in range(0, len(lnS), 500):
        lx = lnS[i0:i0 + 500, None] + sig * z[None, :]
        x = np.exp(np.minimum(lx, 700.0))
        val = np.where(lx < -20, lx - x / 2, np.log(-np.expm1(-np.maximum(x, 1e-300))))
        logh[i0:i0 + 500] = _logsumexp(val + logw, 1)
        log1mh[i0:i0 + 500] = _logsumexp(-x + logw, 1)
    lnt = np.linspace(-45.0, 5.0, 2001)
    dlnt = lnt[1] - lnt[0]
    out = np.empty_like(u)
    for j0 in range(0, len(u), 256):
        ls = lnt[None, :] - u[j0:j0 + 256, None]
        lh = np.where(ls < lo, ls + log_mgf, np.interp(ls, lnS, logh))
        l1 = np.interp(ls, lnS, log1mh)                       # flat beyond hi (<= -40)
        h = np.exp(lh)
        l1 = np.where(h < 0.5, np.log1p(-np.minimum(h, 0.5)), l1)
        integrand = -np.expm1((C - 1) * l1) * np.exp(-np.exp(lnt))[None, :]
        # trapezoid in ln t
        out[j0:j0 + 256] = dlnt * (integrand.sum(1) - 0.5 * (integrand[:, 0] + integrand[:, -1]))
    return out


def shape_loss_sigma(C, m, sig):
    """Exact ``L_C(m, s^2)`` for one ``sig = s / sqrt(2)`` and an array of ``m``: the
    per-class loss of the symmetric state ``q_c - q_k = m + sig (y - z_k)`` (``y, z_k``
    i.i.d. ``N(0, 1)``), i.e. ``O_cc - O_kc = m`` and ``Var(q_c - q_k) = s^2`` for all ``k``."""
    m = np.atleast_1d(np.asarray(m, dtype=float))
    if sig < 1e-7:
        return np.logaddexp(0.0, np.log(C - 1) - m)
    dy = min(0.01, 0.1 / sig)
    y = np.arange(-13.0 - sig, 13.0 + dy / 2, dy)
    logwy = -y ** 2 / 2 - 0.5 * np.log(2 * np.pi) + np.log(dy)
    u_lo, u_hi = m.min() + sig * y[0], m.max() + sig * y[-1]
    nu = int(min(20001, max(401, (u_hi - u_lo) / 0.02)))
    u = np.linspace(u_lo, u_hi, nu)
    logL = np.log(_lambda_table(C, sig, u))
    out = np.empty_like(m)
    for i, mi in enumerate(m):
        out[i] = np.exp(_logsumexp(np.interp(mi + sig * y, u, logL) + logwy, 0))
    return out


def shape_loss(C, m, s2):
    """``L_C(m, s^2)`` for scalars / same-shape arrays (one Frullani table per distinct ``s``;
    use :class:`ShapeTable` for many states)."""
    m, s2 = np.broadcast_arrays(np.asarray(m, dtype=float), np.asarray(s2, dtype=float))
    out = np.empty(m.shape)
    for s in np.unique(s2):
        k = s2 == s
        out[k] = shape_loss_sigma(C, m[k], np.sqrt(max(s, 0.0) / 2))
    return out


class ShapeTable:
    """``log L_C`` on an ``(s, m)`` grid, bilinear interpolation (validated in
    ``recheck_cached.py`` against direct evaluations)."""

    def __init__(self, C, s_max, m_lo, m_hi, ns=64, nm=1201):
        self.C = C
        self.s = np.concatenate([[0.0], np.geomspace(0.02, s_max, ns)])
        self.m = np.linspace(m_lo, m_hi, nm)
        self.logL = np.array([np.log(shape_loss_sigma(C, self.m, s / np.sqrt(2)))
                              for s in self.s])

    def __call__(self, m, s2):
        from scipy.interpolate import RegularGridInterpolator
        f = RegularGridInterpolator((self.s, self.m), self.logL, bounds_error=False,
                                    fill_value=None)
        m, s2 = np.broadcast_arrays(np.asarray(m, float), np.asarray(s2, float))
        pts = np.stack([np.sqrt(np.clip(s2, 0, None)).ravel(), m.ravel()], -1)
        return np.exp(f(pts)).reshape(m.shape)
