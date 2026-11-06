"""Self-tests for series J (``real/bin/python experiments/knee.py selftest``).

1. ``run_mixture_at`` / ``run_mixture_at_many`` are bit-identical to the pinned
   ``0db8cce:simulate.py`` (C = 8, D = 300, 4 keys, with and without bias, Adam and SGD).
2. ``run_mixture_segments`` (U = 1) reproduces ``run_mixture_at`` on the same grid; the
   knee chunk executable (traced priors / lr / beta2 / eps) is compared too.
3. Resume: 6 chunks direct vs 3 chunks + checkpoint (``.npy`` + ``meta.json``) + 3 chunks,
   Adam (vmapped beta2 lanes) and SGD, U in {1, 4}, float64 and float32: ``array_equal`` on
   every chunk file array and on the final checkpoint; plus one resume across processes.
4. U = 4 vs U = 1 agree to 1e-12 relative (float64).
5. A traced-beta2 lane matches a scalar-beta2 simulator (Python-float ``beta2``, ``lr``,
   ``eps``; constant data) to round-off.
6. Diagnostics vs a numpy recompute from the returned carry (C = 4, D = 20): B, O, U, vbar,
   nu, frac_eps, Bdiag, Odiag, margins, Udiag, seed means, loss / err.
7. Keys are prefix-stable in ``runs`` and shared across beta2 lanes; the record grid is
   nested, horizon-independent and chunk ends sit on half decades.
8. Loss MC with 2^14 vs 2^17 draws at late cached series-I states (< 1% or FLAG), for the
   plain estimator (``loss``) and the control-variate one (``loss_cv``).
9. ``certify`` on synthetic records: FIXED, TAIL, MOVING and UNRESOLVED cases.
"""
from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

import knee as J  # noqa: E402

zs, jax, jnp, problems, simulate = J.jax_modules()

PINNED = "0db8cce"
RESULTS = []


def check(name, ok, detail, flag=False):
    tag = "PASS" if ok else ("FLAG" if flag else "FAIL")
    RESULTS.append({"name": name, "pass": bool(ok), "flag": bool(flag and not ok),
                    "detail": detail})
    print(f"[{tag}] {name}: {detail}", flush=True)


def maxdiff(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(np.max(np.abs(a - b))) if a.size else 0.0


def reldiff(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return maxdiff(a, b) / max(float(np.max(np.abs(b))), 1e-300)


def pinned_simulate():
    """``simulate.py`` at the pinned commit (before series J's additions)."""
    src = subprocess.check_output(["git", "show", f"{PINNED}:simulate.py"], cwd=ROOT, text=True)
    path = Path(tempfile.mkdtemp()) / "simulate_pinned.py"
    path.write_text(src)
    spec = importlib.util.spec_from_file_location("simulate_pinned", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# --------------------------------------------------------------------------- #
def test_pinned():
    old = pinned_simulate()
    priors, data, theta0 = zs.zipf_model(8, 2.0, 300)
    problem = problems.make_gmm_softmax(np.asarray(priors))
    old_data = old.MixtureData(*data)
    steps = np.array([0, 10, 100, 1000])
    keys = jax.random.split(jax.random.PRNGKey(3), 4)
    worst, n_arrays = 0.0, 0
    for bias in (False, True):
        p0 = jnp.concatenate([theta0, jnp.zeros((1, 8))]) if bias else theta0
        for b in (1, 4):
            pairs = [
                (simulate.build_mixture_adam(problem, data, 0.3 / 300, beta1=0.9, beta2=0.95,
                                             eps=1e-8, batch=b, bias=bias),
                 old.build_mixture_adam(problem, old_data, 0.3 / 300, beta1=0.9, beta2=0.95,
                                        eps=1e-8, batch=b, bias=bias)),
                (simulate.build_mixture_sgd(problem, data, 3 / 300, batch=b, bias=bias),
                 old.build_mixture_sgd(problem, old_data, 3 / 300, batch=b, bias=bias))]
            for new_sim, old_sim in pairs:
                n = simulate.run_mixture_at_many(new_sim, data, p0, steps, keys=keys, bias=bias)
                o = old.run_mixture_at_many(old_sim, old_data, p0, steps, keys=keys, bias=bias)
                for a, c in zip(n, o):
                    worst = max(worst, maxdiff(a, c))
                    n_arrays += 1
                # single-key driver too
                n1 = simulate.run_mixture_at(new_sim, data, p0, steps, key=keys[1], bias=bias)
                o1 = old.run_mixture_at(old_sim, old_data, p0, steps, key=keys[1], bias=bias)
                for a, c in zip(n1, o1):
                    worst = max(worst, maxdiff(a, c))
                    n_arrays += 1
    check(f"run_mixture_at/_many bit-identical to {PINNED}:simulate.py", worst == 0.0,
          f"max |diff| {worst:g} over {n_arrays} arrays (bias on/off, b in 1,4, Adam, SGD)")


def test_segments_vs_at():
    C, D = 8, 300
    priors, data, theta0 = zs.zipf_model(C, 1.0, D, noise=0.5)
    problem = problems.make_gmm_softmax(np.asarray(priors))
    steps = J.record_steps(D, 6 * J.H)                      # up to t = 10 (3000 steps)
    bounds = np.stack([np.concatenate([[0], steps[:-1]]), steps], 1)
    keys = jax.random.split(jax.random.PRNGKey(11), 3)
    cov = simulate.mixture_covariance(data)

    def obs(params, state):
        W = jnp.concatenate([params, data.teacher], axis=1)
        KW = cov[:, None] * W
        return W.T @ KW, W.T @ data.means

    worst = {}
    for opt in ("adam", "sgd"):
        sim = (simulate.build_mixture_adam(problem, data, 0.3 / D, beta1=0.9, beta2=0.99,
                                           eps=1e-8) if opt == "adam"
               else simulate.build_mixture_sgd(problem, data, 3.0 / D))
        ref = simulate.run_mixture_at_many(sim, data, theta0, steps, keys=keys)
        seg = jax.jit(jax.vmap(lambda k: simulate.run_mixture_segments(
            sim, data, simulate.mixture_carry(sim, theta0, k), bounds, observe=obs)[1]))(keys)
        worst[opt] = max(maxdiff(a, b) for a, b in zip(seg, ref))
    ok = all(v == 0.0 for v in worst.values())
    check("run_mixture_segments (U=1) vs run_mixture_at", ok or max(worst.values()) < 1e-13,
          f"bitwise={ok}; max |diff| adam {worst['adam']:g}, sgd {worst['sgd']:g}")

    # the knee chunk executable (traced priors, lr, beta2, eps) on the same trajectory
    g = J.Group("adam", C, 1.0, 0.5, 1e-8, D, 3, beta2s=(0.99,))
    runner = J.Runner(g)
    # use the same keys as above: overwrite the lane keys
    params, state, _ = runner.init()
    carry = (params, state, keys)
    O_path = []
    for n in range(6):
        carry, rec = runner.advance(carry, n)
        O_path.append(np.asarray(rec["O"]))
    O_path = np.concatenate(O_path, 1)                      # (lanes, 48, C, C)
    sim = simulate.build_mixture_adam(problem, data, 0.3 / D, beta1=0.9, beta2=0.99, eps=1e-8)
    ref = simulate.run_mixture_at_many(sim, data, theta0, steps, keys=keys)
    d = maxdiff(O_path, ref[1])
    r = d / float(np.max(np.abs(np.asarray(ref[1]))))
    check("knee chunk executable vs run_mixture_at (Adam, 6 chunks)", r < 1e-12,
          f"bitwise={d == 0.0}; max |diff| O {d:.3g} (relative {r:.2g})")


def _compare_dirs(da, db, n):
    worst, bad = 0, []
    for k in range(n):
        a, b = dict(np.load(J.chunk_path(da, k))), dict(np.load(J.chunk_path(db, k)))
        for key in a:
            if key.startswith("seconds"):
                continue
            if not np.array_equal(a[key], b[key], equal_nan=a[key].dtype.kind == "f"):
                bad.append(f"chunk{k}:{key}")
    ma, mb = J.read_meta(da), J.read_meta(db)
    for f in ("params.npy", "m.npy", "v.npy", "key.npy", "step.npy", "record_index.npy"):
        pa, pb = Path(da) / ma["ckpt"] / f, Path(db) / mb["ckpt"] / f
        if pa.exists() != pb.exists():
            bad.append(f"ckpt:{f} presence")
        elif pa.exists() and not np.array_equal(np.load(pa), np.load(pb)):
            bad.append(f"ckpt:{f}")
    return bad


def test_resume(tmp):
    groups = {}
    for opt in ("adam", "sgd"):
        for dtype in ("f64", "f32"):
            for U in (1, 4):
                g = J.Group(opt, 8, 1.0, 0.5, 1e-8, 300, 3, dtype=dtype, unroll=U,
                            beta2s=(0.9, 0.99))
                da, db = tmp / "direct", tmp / "resumed"
                J.advance_group(g, 6, root=da, quiet=True)
                J.advance_group(g, 3, root=db, quiet=True)
                J.advance_group(g, 6, root=db, quiet=True)      # reopens from disk
                bad = _compare_dirs(J.group_dir(g, da), J.group_dir(g, db), 6)
                check(f"resume bit-identity {opt} {dtype} U={U}", not bad,
                      "6 chunks direct == 3 + checkpoint + 3 (all chunk arrays and final"
                      " checkpoint)" if not bad else f"differs: {bad[:6]}")
                groups[(opt, dtype, U)] = J.group_dir(g, da)
    return groups


def test_resume_subprocess(tmp):
    """Chunks 0-2 in a separate process (fresh compile), chunks 3-5 here."""
    g = J.Group("adam", 8, 1.0, 0.5, 1e-8, 300, 3, beta2s=(0.9, 0.99))
    root = tmp / "subproc"
    code = (f"import sys; sys.path.insert(0, {str(ROOT / 'experiments')!r}); import knee as J;"
            f" g = J.Group('adam', 8, 1.0, 0.5, 1e-8, 300, 3, beta2s=(0.9, 0.99));"
            f" J.advance_group(g, 3, root={str(root)!r}, quiet=True)")
    t0 = time.time()
    subprocess.run([sys.executable, "-c", code], check=True, cwd=ROOT,
                   stderr=subprocess.DEVNULL)
    J.advance_group(g, 6, root=root, quiet=True)
    bad = _compare_dirs(J.group_dir(g, tmp / "direct"), J.group_dir(g, root), 6)
    check("resume across processes (Adam f64 U=1)", not bad,
          f"identical to the direct run ({time.time() - t0:.0f}s)" if not bad
          else f"differs: {bad[:6]}")


def test_unroll(groups):
    out = {}
    for opt in ("adam", "sgd"):
        for dtype in ("f64", "f32"):
            a = J.load_group(groups[(opt, dtype, 1)])
            b = J.load_group(groups[(opt, dtype, 4)])
            r = max(reldiff(b[k], a[k]) for k in ("B_mean", "O_mean", "Bdiag", "Odiag",
                                                    "loss"))
            ca = np.load(Path(groups[(opt, dtype, 1)]) / J.read_meta(groups[(opt, dtype, 1)])["ckpt"] / "params.npy")
            cb = np.load(Path(groups[(opt, dtype, 4)]) / J.read_meta(groups[(opt, dtype, 4)])["ckpt"] / "params.npy")
            out[(opt, dtype)] = (max(r, reldiff(cb, ca)), maxdiff(cb, ca) == 0.0)
    for opt in ("adam", "sgd"):
        r, bit = out[(opt, "f64")]
        check(f"U=4 vs U=1 {opt} f64", r < 1e-12, f"max relative diff {r:.2g} (bitwise {bit})")
    print(f"      (info) f32: adam {out[('adam', 'f32')][0]:.2g} (bitwise "
          f"{out[('adam', 'f32')][1]}), sgd {out[('sgd', 'f32')][0]:.2g} (bitwise "
          f"{out[('sgd', 'f32')][1]})", flush=True)
    return out


def test_traced_beta2():
    C, D, runs = 8, 300, 2
    g = J.Group("adam", C, 1.0, 0.5, 1e-8, D, runs, beta2s=(0.9, 0.99))
    runner = J.Runner(g)
    carry = runner.init()
    for n in range(6):
        carry, _ = runner.advance(carry, n)
    priors, data, theta0 = zs.zipf_model(C, 1.0, D, noise=0.5)
    problem = problems.make_gmm_softmax(np.asarray(priors))
    keys = J.run_keys(runs)
    worst, bit = 0.0, True
    for b, b2 in enumerate((0.9, 0.99)):
        sim = simulate.build_mixture_adam(problem, data, 0.3 / D, beta1=0.9, beta2=b2, eps=1e-8)
        obs = lambda p, s: p[:1, :1]

        @jax.jit
        def ref(c, bounds):
            return jax.vmap(lambda cc: simulate.run_mixture_segments(sim, data, cc, bounds,
                                                                     observe=obs))(c)
        c = jax.vmap(lambda k: simulate.mixture_carry(sim, theta0, k))(keys)
        for n in range(6):
            c, _ = ref(c, jnp.asarray(J.chunk_bounds(D, n)))
        lanes = slice(b * runs, (b + 1) * runs)
        for x, y in ((carry[0][lanes], c[0]), (carry[1][0][lanes], c[1][0]),
                     (carry[1][1][lanes], c[1][1])):
            worst = max(worst, reldiff(x, y))
            bit &= maxdiff(x, y) == 0.0
        bit &= bool(np.array_equal(np.asarray(carry[2][lanes]), np.asarray(c[2])))
    check("traced-beta2 lane vs scalar-beta2 simulator", worst < 1e-12,
          f"max relative diff (params, m, v) {worst:.2g}; bitwise {bit}")


def test_diagnostics():
    C, D, runs, eps = 4, 20, 3, 1e-3
    g = J.Group("adam", C, 1.0, 0.5, eps, D, runs, beta2s=(0.9, 0.999))
    runner = J.Runner(g)
    carry = runner.init()
    for n in range(6):
        carry, _ = runner.advance(carry, n)
    carry, rec = runner.advance(carry, 6)
    red = {k: np.asarray(v) for k, v in runner.reduce(rec).items()}
    params, (m, v), _ = (np.asarray(carry[0]), tuple(np.asarray(x) for x in carry[1]), None)
    priors, data, theta0 = zs.zipf_model(C, 1.0, D, noise=0.5)
    K = np.asarray(data.factor) ** 2
    means = np.asarray(data.means)
    errs = {}
    L = g.lanes
    Bn = np.einsum("lic,i,lid->lcd", params, K, params)
    On = np.einsum("lic,id->lcd", params, means)
    stab = eps * K[None, :, None]
    den = np.sqrt(v + stab)
    Un = np.einsum("ic,lik->lck", means, m / den)
    vbar, nu, frac = v.mean(1), (np.abs(m) / den).mean(1), (v < stab).mean(1)
    last = {k: np.asarray(x)[:, -1] for k, x in rec.items()}
    for k, ref in (("B", Bn), ("O", On), ("U", Un), ("vbar", vbar), ("nu", nu),
                   ("frac_eps", frac)):
        errs[k] = reldiff(last[k], ref)
    per = lambda x: x.reshape((2, runs) + x.shape[1:])
    Od = np.diagonal(On, axis1=-2, axis2=-1)
    off = On.copy()
    idx = np.arange(C)
    off[:, idx, idx] = np.nan
    checks = {"Bdiag": np.diagonal(Bn, axis1=-2, axis2=-1), "Odiag": Od,
              "margin_mean": Od - np.nanmean(off, 1), "margin_min": Od - np.nanmax(off, 1),
              "Udiag": np.diagonal(Un, axis1=-2, axis2=-1), "vbar": vbar, "nu": nu,
              "frac_eps": frac}
    for k, ref in checks.items():
        errs["red_" + k] = reldiff(red[k][-1], per(ref))
    errs["B_mean"] = reldiff(red["B_mean"][-1], per(Bn).mean(1))
    errs["O_mean"] = reldiff(red["O_mean"][-1], per(On).mean(1))
    u32 = reldiff(red["U_mean"][-1], per(Un).mean(1))
    loss, err = [], []
    for l in range(L):
        a, b = problems.softmax_class_metrics(jnp.asarray(Bn[l]), jnp.asarray(On[l]),
                                              samples=J.SAMPLES)
        loss.append(np.asarray(a))
        err.append(np.asarray(b))
    errs["loss"] = reldiff(red["loss"][-1], per(np.array(loss)))
    lcv = [np.asarray(J.class_metrics(jnp.asarray(Bn[l]), jnp.asarray(On[l]))[2])
           for l in range(L)]
    errs["loss_cv"] = reldiff(red["loss_cv"][-1], per(np.array(lcv)))
    cv_vs_mc = reldiff(red["loss_cv"][-1], red["loss"][-1])
    errs["err"] = maxdiff(red["err"][-1], per(np.array(err)))
    worst = max(errs.values())
    frac_range = (float(frac.min()), float(frac.max()))
    check("diagnostics vs numpy recompute (C=4, D=20, eps=1e-3)", worst < 1e-12 and u32 < 1e-6,
          f"worst relative diff {worst:.2g} ({max(errs, key=errs.get)}); U_mean (f32) {u32:.2g};"
          f" frac_eps range {frac_range}; loss_cv vs loss {cv_vs_mc:.2g}")


def test_keys_grid():
    a, b = np.asarray(J.run_keys(4)), np.asarray(J.run_keys(9))
    prefix = np.array_equal(a, b[:4])
    g = J.Group("adam", 8, 1.0, 0.5, 1e-8, 300, 4, beta2s=(0.9, 0.95, 0.99))
    keys = np.asarray(J.init_carry(g, jnp.zeros((300, 8)))[2])
    lanes_ok = all(np.array_equal(keys[bb * 4:(bb + 1) * 4], a) for bb in range(3))
    distinct = len({tuple(k) for k in a}) == 4
    check("keys prefix-stable in runs, shared across beta2 lanes", prefix and lanes_ok and distinct,
          f"prefix {prefix}, lanes {lanes_ok}, distinct {distinct}")
    ok, info = True, []
    for D in (20, 300, 1000, 2000, 5000):
        long = J.record_steps(D, 200)
        ok &= np.array_equal(J.record_steps(D, 40), long[:40])
        ok &= bool(np.all(np.diff(long) >= 0))
        for n in range(12):
            bnd = J.chunk_bounds(D, n)
            j = J.chunk_records(n)
            ok &= np.array_equal(bnd[:, 1], long[j])
            ok &= np.array_equal(bnd[1:, 0], bnd[:-1, 1])
            if n:
                ok &= bnd[0, 0] == J.chunk_bounds(D, n - 1)[-1, 1]
            ok &= abs(bnd[-1, 1] / D - 0.01 * 10 ** (n / 2)) <= 0.5 / D + 1e-12
        info.append(int((np.diff(long[:40]) == 0).sum()))
    ok &= all(J.chunks_for(0.01 * 10 ** (n / 2)) == n + 1 for n in range(1, 16))
    check("record grid nested, horizon-independent, chunk ends on half decades", bool(ok),
          f"zero-length segments in the first 40 records for D=20,300,1000,2000,5000: {info}")


def test_mc_draws():
    ode = ROOT / "figs" / "experiments" / "imbalance" / "ode"
    names = ("m0_adam_k0.1_C128_a1_b10.9_b20.999_b1_e1e-08_eta0.3_T1000_s1.npz",
             "m0_adam_k0.1_C128_a2_b10.9_b20.95_b1_e1e-08_eta0.3_T316.228_s1.npz",
             "m0_sgd_k0.1_C128_a1_b1_eta3_T10000_s1.npz",
             "m0_adam_k1_C128_a1_b10.9_b20.95_b1_e1e-08_eta0.3_T316.228_s1.npz")
    files = [ode / n for n in names if (ode / n).exists()]
    if not files:
        check("metrics 2^14 vs 2^17 draws", False, "no cached late state found", flag=True)
        return
    fn = {s: jax.jit(lambda B, O, s=s: problems.softmax_class_metrics(B, O, samples=s))
          for s in (2 ** 14, 2 ** 17)}
    cv = {s: jax.jit(lambda B, O, s=s: J.class_metrics(B, O, samples=s)[2])
          for s in (2 ** 14, 2 ** 17)}
    fused = jax.jit(lambda B, O: J.class_metrics(B, O, samples=2 ** 14))
    dflt = jax.jit(lambda B, O: J.class_metrics(B, O, samples=J.SAMPLES, seed=1))
    ref = jax.jit(lambda B, O: J.class_metrics(B, O, samples=2 ** 19, seed=2))
    for f in files:
        d = np.load(f)
        B, O = jnp.asarray(d["B"][-1]), jnp.asarray(d["O"][-1])
        l14, e14 = (np.asarray(x) for x in fn[2 ** 14](B, O))
        l17, e17 = (np.asarray(x) for x in fn[2 ** 17](B, O))
        c14, c17 = np.asarray(cv[2 ** 14](B, O)), np.asarray(cv[2 ** 17](B, O))
        fl, fe, _ = (np.asarray(x) for x in fused(B, O))
        same_rel = float(np.max(np.abs(fl - l14) / l14))
        check(f"class_metrics == softmax_class_metrics (2^14) at {f.name[:32]}...",
              same_rel < 1e-6 and np.array_equal(fe, e14),
              f"loss max rel diff {same_rel:.2g} (abs {np.max(np.abs(fl - l14)):.2g}; the "
              f"logsumexp - q_c form cancels at small loss), err identical "
              f"{np.array_equal(fe, e14)}")
        rel = np.abs(l14 - l17) / l17
        rel_cv = np.abs(c14 - c17) / c17
        worst, worst_cv = float(rel.max()), float(rel_cv.max())
        tag = f"T={d['time'][-1]:g} ({f.name[:32]}...)"
        check(f"loss MC 2^14 vs 2^17 at {tag}", worst < 0.01,
              f"max rel diff {worst:.3g} (median {np.median(rel):.2g}; loss range "
              f"{l17.min():.2g}..{l17.max():.2g}); err max |diff| "
              f"{np.abs(e14 - e17).max():.2g}", flag=True)
        d0, r0 = (np.asarray(x) for x in dflt(B, O)), (np.asarray(x) for x in ref(B, O))
        (dl, _, dc), (rl, _, rc) = d0, r0
        rd, rdc = np.abs(dl - rl) / rl, np.abs(dc - rc) / rc
        check(f"default SAMPLES=2^{int(np.log2(J.SAMPLES))} vs 2^19 (independent draws) at {tag}",
              float(rd.max()) < 0.03,
              f"loss max rel {rd.max():.3g} (median {np.median(rd):.2g}); loss_cv max rel "
              f"{rdc.max():.3g} (median {np.median(rdc):.2g})", flag=True)
        check(f"loss_cv 2^14 vs 2^17 at {tag}", worst_cv < 0.01,
              f"max rel diff {worst_cv:.3g} (median {np.median(rel_cv):.2g}); "
              f"loss_cv(2^17) vs loss(2^17) max rel {float(np.max(np.abs(c17 - l17) / l17)):.3g}",
              flag=True)


def test_certify():
    rng = np.random.default_rng(1)
    t = np.concatenate([[0.0], J.T_MIN * 10 ** (np.arange(-6, 8 * 8 + 1) / 16)])  # to T=100
    N, nb, runs, C = len(t), 1, 8, 4
    tt = np.maximum(t, 1e-3)[:, None, None, None]
    noise = lambda s: 1 + s * rng.standard_normal((N, nb, runs, C))
    # class 0: FIXED (plateau after t = 1), 1: TAIL (1/t from t = 0.5), 2: MOVING (exp(-t/30)),
    # 3: TAIL-like start but slope still changing (t^-0.5 -> t^-1 crossover at t = 30)
    base = np.ones((N, nb, runs, C))
    base[..., 0] = 0.5 + 0.5 * np.exp(-tt[..., 0] / 0.3)
    base[..., 1] = np.minimum(1.0, 0.5 / tt[..., 0])
    base[..., 2] = np.exp(-tt[..., 0] / 30)
    base[..., 3] = 1 / (1 + (tt[..., 0] / 0.1) ** 0.5 * (1 + tt[..., 0] / 30) ** 0.5)
    loss = base * noise(1e-3)
    # gaps: stationary for class 0, growing ~ log t for the others
    O = np.zeros((N, nb, C, C))
    grow = np.log1p(t)[:, None]
    for c in range(C):
        O[:, :, c, c] = 3.0 if c == 0 else grow
    O[:, :, 0, 0] = 3.0 * (1 - np.exp(-np.maximum(t, 0)[:, None] / 0.3))
    mm = np.repeat(np.diagonal(O, axis1=-2, axis2=-1)[:, :, None, :], runs, 2) * noise(1e-4)
    rec = {"time": t, "loss": loss, "margin_mean": mm, "O_mean": O,
           "frac_eps": np.zeros_like(loss)}
    cert = J.certify(rec)
    got = [str(x) for x in cert["status"][0]]
    want = ["FIXED", "TAIL", "MOVING", "MOVING"]
    cap = J.certify(rec, at_cap=True)["status"][0][2]
    check("certify on synthetic records", got == want and cap == "UNRESOLVED",
          f"status {got} (want {want}); at cap class 2 -> {cap}; slopes last decade "
          f"{np.round(cert['slope_last_decade'][0], 3).tolist()}, prev "
          f"{np.round(cert['slope_prev_decade'][0], 3).tolist()}")


def main(quick=False):
    t0 = time.time()
    tmp = Path(tempfile.mkdtemp(prefix="knee_selftest_"))
    try:
        test_keys_grid()
        test_certify()
        test_pinned()
        test_segments_vs_at()
        groups = test_resume(tmp)
        test_unroll(groups)
        if not quick:
            test_resume_subprocess(tmp)
        test_traced_beta2()
        test_diagnostics()
        test_mc_draws()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    fails = [r for r in RESULTS if not r["pass"] and not r["flag"]]
    flags = [r for r in RESULTS if r["flag"]]
    print(f"\n{len(RESULTS) - len(fails) - len(flags)} passed, {len(fails)} failed, "
          f"{len(flags)} flagged ({time.time() - t0:.0f}s)", flush=True)
    out = J.OUT / "selftest.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"results": RESULTS, "git": J.git_hash(),
                               "fingerprint": J.numerics_fingerprint(),
                               "seconds": time.time() - t0}, indent=1))
    if fails:
        sys.exit(1)


if __name__ == "__main__":
    main(quick="--quick" in sys.argv)
