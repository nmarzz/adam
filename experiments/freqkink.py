"""Frequency kink in Adam: linear softmax head on frozen GPT-2 embeddings.

Plan: ``experiments/Frequency kink in Adam logistic regression on pretrained
embeddings.md``. Data from ``experiments/freqkink_prep.py`` (``data/freqkink``).

A linear head W (D x C, no bias, C = 50257) is trained from zero on next-token
prediction with cross-entropy, by one streaming pass over a fixed global
permutation of the training positions (identical for every run, so runs differ
only in the optimizer). Adam has beta1 = 0, bias correction on v, and the usual
``g / (sqrt(v_hat) + eps)``. SGD is the control.

At log-spaced steps the held-out stream is evaluated and per-token sums of the
loss are stored, so per-token and per-frequency-bin curves l_c(t) can be formed
afterwards. For Adam the per-output RMS and median over coordinates of
sqrt(v_hat) are stored as well (to compare against eps). Per-token sums of the
own logit on the held-out contexts and the logits on the mean held-out context
give the class-specific signal S_c = w_c . (xbar_c - xbar), which is insensitive
to the background push of every column along the shared mean direction.

Usage (from the repository root):
    real/bin/python experiments/freqkink.py tune --opt adam --b 32 --beta2 0.99 --eps 1e-12
    real/bin/python experiments/freqkink.py run  --opt adam --b 32 --beta2 0.99 --eps 1e-12 --lr 1e-3
Several configurations can be run in one process (the data are loaded once):
    real/bin/python experiments/freqkink.py sweep --spec results/freqkink/first.json
"""
from __future__ import annotations

import argparse
import json
import os
import time
from functools import partial
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("FREQKINK_DATA", ROOT / "data" / "freqkink"))
OUT = Path(os.environ.get("FREQKINK_OUT", ROOT / "results" / "freqkink"))
C = 50257
EVAL_CHUNK = 16_000


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
@partial(jax.jit, donate_argnums=0)
def _put(buf, chunk, start):
    return lax.dynamic_update_slice(buf, chunk, (start, 0))


def to_device(path, n=None, chunk=1 << 20):
    """Load an (n, D) float16 .npy onto the GPU without a host copy of the whole."""
    X = np.load(path, mmap_mode="r")
    n = len(X) if n is None else n
    buf = jnp.zeros((n, X.shape[1]), jnp.float16)
    for a in range(0, n, chunk):
        c = np.ascontiguousarray(X[a:min(a + chunk, n)])
        buf = _put(buf, jnp.asarray(c), a)
    return buf


class Data:
    def __init__(self, n_train=None, n_test=None, seed=0, center=False):
        t0 = time.time()
        ytr = np.load(DATA / "train_y.npy")
        yte = np.load(DATA / "test_y.npy")
        n_train = len(ytr) if n_train is None else n_train
        n_test = len(yte) if n_test is None else n_test
        n_test -= n_test % EVAL_CHUNK
        self.n_train, self.n_test = n_train, n_test
        self.Xtr = to_device(DATA / "train_X.npy", n_train)
        self.ytr = jnp.asarray(ytr[:n_train])
        self.Xte = to_device(DATA / "test_X.npy", n_test).reshape(-1, EVAL_CHUNK, 768)
        self.yte = jnp.asarray(yte[:n_test]).reshape(-1, EVAL_CHUNK)
        # Optional centring by the training mean (applied on the fly to every batch).
        self.center = center
        if center:
            tot = jnp.zeros(768, jnp.float32)
            for a in range(0, n_train, 1 << 20):
                tot = tot + jnp.sum(self.Xtr[a:a + (1 << 20)], axis=0, dtype=jnp.float32)
            self.mu = tot / n_train
        else:
            self.mu = jnp.zeros(768, jnp.float32)
        self.xbar = jnp.mean(self.Xte, axis=(0, 1), dtype=jnp.float32) - self.mu
        self.perm = jnp.asarray(np.random.default_rng(seed).permutation(n_train).astype(np.int32))
        self.train_counts = np.bincount(ytr[:n_train], minlength=C)
        self.test_counts = np.bincount(yte[:n_test], minlength=C)
        print(f"data: train {n_train:,}  test {n_test:,}  centre={center}  "
              f"|mu|={float(jnp.linalg.norm(self.mu)):.1f}  ({time.time() - t0:.0f}s)", flush=True)


# --------------------------------------------------------------------------- #
# Training and evaluation
# --------------------------------------------------------------------------- #
def make_train(opt, b, beta2, eps):
    """Return a jitted function advancing (W, v) from step t0 to t1."""

    def body(t, state, X, Y, perm, mu, lr):
        W, v = state
        idx = lax.dynamic_slice(perm, (t * b,), (b,))
        x = X[idx].astype(jnp.float32) - mu
        logits = x @ W
        p = jax.nn.softmax(logits, axis=-1)
        G = (p - jax.nn.one_hot(Y[idx], C, dtype=jnp.float32)) / b
        g = x.T @ G
        if opt == "sgd":
            return W - lr * g, v
        v = beta2 * v + (1 - beta2) * g * g
        vhat = v / (1 - beta2 ** (t + 1).astype(jnp.float32))
        return W - lr * g / (jnp.sqrt(vhat) + eps), v

    @partial(jax.jit, donate_argnums=(0, 1))
    def train(W, v, X, Y, perm, mu, t0, t1, lr):
        return lax.fori_loop(t0, t1, lambda t, s: body(t, s, X, Y, perm, mu, lr), (W, v))

    return train


@jax.jit
def evaluate(W, Xte, yte, mu):
    """Per-token sums of held-out cross-entropy and the total."""

    def chunk(carry, xy):
        x, y = xy
        logits = (x.astype(jnp.float32) - mu) @ W
        lse = jax.nn.logsumexp(logits, axis=-1)
        own = jnp.take_along_axis(logits, y[:, None], axis=1)[:, 0]
        L, O = carry
        return (L + jax.ops.segment_sum(lse - own, y, num_segments=C),
                O + jax.ops.segment_sum(own, y, num_segments=C)), None

    z = jnp.zeros(C, jnp.float32)
    (per_tok, own), _ = lax.scan(chunk, (z, z), (Xte, yte))
    return per_tok, own


@jax.jit
def mean_context_logits(W, xbar):
    """Logits of every output on the mean held-out context, W^T xbar."""
    return xbar @ W


@jax.jit
def v_stats(v, t, beta2):
    vhat = v / (1 - beta2 ** t)
    s = jnp.sqrt(vhat)
    return jnp.sqrt(jnp.mean(vhat, axis=0)), jnp.median(s, axis=0)


def schedule(T, per_octave=4):
    """Log-spaced checkpoint steps 0, 1, ..., T."""
    k = np.unique(np.round(np.geomspace(1, T, int(per_octave * np.log2(max(T, 2))) + 1)).astype(int))
    return np.concatenate([[0], k[k <= T]])


def run(data, opt, b, lr, beta2=0.99, eps=1e-12, budget=None, per_octave=4, vstats=True, log=True):
    budget = data.n_train if budget is None else budget
    T = budget // b
    ckpt = np.array([T]) if per_octave == 0 else schedule(T, per_octave)
    train = make_train(opt, b, beta2, eps)
    W = jnp.zeros((768, C), jnp.float32)
    v = jnp.zeros((768, C), jnp.float32) if opt == "adam" else jnp.zeros((1, 1), jnp.float32)
    per_tok, own, gen, vrms, vmed, wall = [], [], [], [], [], []
    t0 = time.time()
    prev = 0
    for t in ckpt:
        if t > prev:
            W, v = train(W, v, data.Xtr, data.ytr, data.perm, data.mu, prev, int(t), jnp.float32(lr))
            prev = int(t)
        pt, ow = evaluate(W, data.Xte, data.yte, data.mu)
        pt = np.asarray(pt)
        per_tok.append(pt); own.append(np.asarray(ow))
        gen.append(np.asarray(mean_context_logits(W, data.xbar)))
        if opt == "adam" and vstats and t > 0:
            r, m = v_stats(v, jnp.float32(t), jnp.float32(beta2))
            vrms.append(np.asarray(r)); vmed.append(np.asarray(m))
        wall.append(time.time() - t0)
        total = pt.sum() / data.n_test
        if log and (t == ckpt[-1] or len(per_tok) % 10 == 0):
            print(f"    step {t:>9,} ({t * b:>11,} samples)  loss {total:.4f}  [{wall[-1]:.0f}s]", flush=True)
        if not np.isfinite(total):
            print("    diverged", flush=True)
            break
    out = dict(steps=ckpt[:len(per_tok)], samples=ckpt[:len(per_tok)] * b,
               per_tok_sum=np.stack(per_tok).astype(np.float32),
               own_logit_sum=np.stack(own).astype(np.float32),
               mean_ctx_logit=np.stack(gen).astype(np.float32),
               total=np.stack(per_tok).sum(1) / data.n_test, wall=np.array(wall))
    if vrms:
        out["v_rms"] = np.stack(vrms).astype(np.float32)
        out["v_med"] = np.stack(vmed).astype(np.float32)
    return out


def tag(opt, b, lr, beta2, eps, center=False):
    s = f"{opt}_b{b}_lr{lr:.3g}"
    s += f"_beta2{beta2}_eps{eps:.0e}" if opt == "adam" else ""
    return s + ("_center" if center else "")


def save(name, out, cfg, data):
    path = OUT / "runs" / f"{name}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    cfg = dict(cfg, center=data.center)
    np.savez_compressed(path, **out, train_counts=data.train_counts, test_counts=data.test_counts,
                        n_test=data.n_test, cfg=json.dumps(cfg))
    print(f"    saved {path}", flush=True)


# --------------------------------------------------------------------------- #
# Step-size tuning: short budget, final total held-out loss
# --------------------------------------------------------------------------- #
def tune(data, opt, b, beta2, eps, grid, budget, max_extend=4):
    """Grid search on the short budget; the grid is extended by factors of
    sqrt(10) while the best value sits at its edge."""
    res = {}

    def try_lr(lr):
        o = run(data, opt, b, lr, beta2, eps, budget=budget, per_octave=0, vstats=False, log=False)
        res[lr] = float(o["total"][-1]) if np.isfinite(o["total"][-1]) else float("inf")
        print(f"    tune {opt} b={b} eps={eps:.0e} lr={lr:.3g}: loss {res[lr]:.4f}", flush=True)

    for lr in grid:
        try_lr(lr)
    for _ in range(max_extend):
        lrs = sorted(res)
        best = min(res, key=res.get)
        if best == lrs[-1]:
            try_lr(float(f"{lrs[-1] * 10 ** 0.5:.3g}"))
        elif best == lrs[0]:
            try_lr(float(f"{lrs[0] / 10 ** 0.5:.3g}"))
        else:
            break
    best = min(res, key=res.get)
    return best, res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "tune", "sweep"])
    ap.add_argument("--opt", default="adam", choices=["adam", "sgd"])
    ap.add_argument("--b", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--beta2", type=float, default=0.99)
    ap.add_argument("--eps", type=float, default=1e-12)
    ap.add_argument("--budget", type=int, default=None)
    ap.add_argument("--n-train", type=int, default=None)
    ap.add_argument("--n-test", type=int, default=None)
    ap.add_argument("--spec", type=str, default=None)
    ap.add_argument("--center", action="store_true", help="subtract the training mean")
    args = ap.parse_args()
    spec = json.loads(Path(args.spec).read_text()) if args.spec else {}
    data = Data(args.n_train, args.n_test, center=args.center or spec.get("center", False))

    if args.cmd == "run":
        cfg = dict(opt=args.opt, b=args.b, lr=args.lr, beta2=args.beta2, eps=args.eps)
        out = run(data, **cfg, budget=args.budget)
        save(tag(**cfg, center=data.center), out, cfg, data)
    elif args.cmd == "tune":
        grid = ADAM_GRID if args.opt == "adam" else SGD_GRID
        tune(data, args.opt, args.b, args.beta2, args.eps, grid, args.budget or data.n_train // 10)
    else:
        sweep(data, spec)


ADAM_GRID = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2]
SGD_GRID = [1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4]


def sweep(data, spec):
    """spec: {"tune_budget": int, "center": bool, "configs": [{opt, b, beta2, eps}, ...]}.

    For each configuration: tune lr on the short budget (results cached in
    tune.json), then run the full pass at the best lr.
    """
    tfile = OUT / "tune.json"
    tuned = json.loads(tfile.read_text()) if tfile.exists() else {}
    for cfg in spec["configs"]:
        cfg = dict(beta2=0.99, eps=1e-12, **cfg)
        key = tag(cfg["opt"], cfg["b"], 0, cfg["beta2"], cfg["eps"], data.center).replace("_lr0", "")
        print(f"=== {key}", flush=True)
        if key not in tuned:
            grid = cfg.pop("grid", ADAM_GRID if cfg["opt"] == "adam" else SGD_GRID)
            best, res = tune(data, cfg["opt"], cfg["b"], cfg["beta2"], cfg["eps"], grid,
                             spec["tune_budget"])
            if best in (grid[0], grid[-1]):
                print(f"    WARNING: best lr {best} at the edge of the grid", flush=True)
            tuned[key] = {"best": best, "losses": {f"{k:.3g}": v for k, v in res.items()}}
            tfile.write_text(json.dumps(tuned, indent=1))
        cfg.pop("grid", None)
        lr = cfg.pop("lr", None) or tuned[key]["best"]
        name = tag(cfg["opt"], cfg["b"], lr, cfg["beta2"], cfg["eps"], data.center)
        if (OUT / "runs" / f"{name}.npz").exists():
            print(f"    {name} exists, skipping", flush=True)
            continue
        c = dict(cfg, lr=lr)
        out = run(data, **c)
        save(name, out, c, data)


if __name__ == "__main__":
    main()
