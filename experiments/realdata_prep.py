"""Build real-data feature sets for the single-Gaussian Adam ODE (scalar regression).

Each builder writes ``data/processed/<name>.npz`` with

* ``X`` (n, D) float32 features, columns centred,
* ``y`` (n,) float32 targets, centred,
* ``info`` a short description string.

The population is the empirical distribution of the n rows: K = X^T X / n,
theta* = K^{-1} X^T y / n and omega = y - X theta* (computed in realdata.py).

Raw files are read from ``data/raw/<dataset>/`` (downloaded separately) and only
parsed with numpy/zipfile/pickle of the official CIFAR batches.

Usage (from the repository root):
    real/bin/python experiments/realdata_prep.py cifar --feat sketch --D 2048
    real/bin/python experiments/realdata_prep.py owt --feat raw
    real/bin/python experiments/realdata_prep.py msd --feat relu --D 1024
"""
from __future__ import annotations

import argparse
import io
import pickle
import tarfile
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed"
POWER = Path.home() / "Documents" / "power" / "embed_datasets"


# --------------------------------------------------------------------------- #
# Feature maps
# --------------------------------------------------------------------------- #
def sphere(Z):
    """Centre each row and scale it to unit norm (thin-shell preprocessing)."""
    Z = Z - Z.mean(axis=1, keepdims=True)
    return Z / np.linalg.norm(Z, axis=1, keepdims=True)


def standardize_cols(Z):
    Z = Z - Z.mean(axis=0)
    return Z / (Z.std(axis=0) + 1e-12)


def features(Z, kind, D, seed, chunk=65_536):
    """Map raw inputs ``Z`` (n, p) to features (n, D).

    sketch: Gaussian sketch S Z with S_{ij} ~ N(0, 1/p) (linear).
    relu / tanh: random features sigma(W Z) with W_{ij} ~ N(0, 1/p) * scale,
        followed by an independent Gaussian sketch (``kind`` = relu+sketch).
    raw: Z itself.
    """
    if kind == "raw":
        return Z.astype(np.float32)
    rng = np.random.default_rng(seed)
    n, p = Z.shape
    base = kind.split("+")[0]
    post = kind.endswith("+sketch")
    W = rng.standard_normal((p, D)).astype(np.float32) / np.sqrt(p)
    S = rng.standard_normal((D, D)).astype(np.float32) / np.sqrt(D) if post else None
    # Rows of Z have norm ~ sqrt(p) after standardization, ~1 after sphere();
    # put pre-activations at unit scale.
    scale = 1.0 / np.sqrt(np.mean(np.sum(Z[:4096] ** 2, axis=1)) / p)
    out = np.empty((n, D), np.float32)
    for a in range(0, n, chunk):
        h = Z[a:a + chunk].astype(np.float32) @ W * scale
        if base == "relu":
            h = np.maximum(h, 0.0)
        elif base == "tanh":
            h = np.tanh(h)
        elif base != "sketch":
            raise ValueError(kind)
        if post:
            h = h @ S
        out[a:a + chunk] = h
    return out


def finish(name, X, y, info):
    X = X - X.mean(axis=0, dtype=np.float64).astype(np.float32)
    y = (y - y.mean()).astype(np.float32)
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{name}.npz"
    np.savez(path, X=X, y=y, info=info)
    print(f"wrote {path}  X {X.shape}  y std {y.std():.4g}  [{info}]")


# --------------------------------------------------------------------------- #
# Datasets
# --------------------------------------------------------------------------- #
CIFAR_ANIMALS = (2, 3, 5)     # bird, cat, dog
CIFAR_VEHICLES = (0, 1, 8, 9)  # plane, car, ship, truck


def load_cifar():
    path = RAW / "cifar10" / "cifar-10-python.tar.gz"
    xs, ys = [], []
    with tarfile.open(path) as tar:
        for member in tar.getmembers():
            base = Path(member.name).name
            if base.startswith("data_batch") or base == "test_batch":
                d = pickle.load(tar.extractfile(member), encoding="bytes")
                xs.append(np.asarray(d[b"data"], np.float32) / 255.0)
                ys.append(np.asarray(d[b"labels"]))
    return np.concatenate(xs), np.concatenate(ys)


def build_cifar(args):
    Z, labels = load_cifar()
    if args.target == "animal":
        y = np.where(np.isin(labels, CIFAR_ANIMALS), 1.0,
                     np.where(np.isin(labels, CIFAR_VEHICLES), -1.0, 0.0))
        keep = y != 0
    else:  # all ten classes, +-1 for animals (2-7) vs vehicles
        y = np.where(np.isin(labels, (0, 1, 8, 9)), -1.0, 1.0)
        keep = np.ones_like(y, bool)
    Z, y = sphere(Z[keep]), y[keep]
    X = features(Z, args.feat, args.D, args.seed)
    name = args.name or f"cifar_{args.target}_{args.feat}_{X.shape[1]}"
    finish(name, X, y, f"CIFAR-10 {args.target} +-1, sphere, {args.feat} D={X.shape[1]}")


def load_owt(n):
    E = np.load(POWER / "openweb_embeddings.npy", mmap_mode="r")
    T = np.load(POWER / "openweb_targets.npy")
    return np.asarray(E[:n], np.float32), T[:n]


def build_owt(args):
    """GPT-2 final hidden states on OpenWebText; target = log unigram frequency
    of the next token (frequencies counted on the same 1M positions)."""
    n = args.n or 1_000_205
    Z, tok = load_owt(n)
    counts = np.bincount(tok)
    y = np.log(counts[tok] / counts.sum())
    if args.standardize:
        Z = standardize_cols(Z)
    if args.sphere:
        Z = sphere(Z)
    X = features(Z, args.feat, args.D, args.seed)
    tag = ("z" if args.standardize else "") + ("s" if args.sphere else "")
    name = args.name or f"owt_logfreq_{args.feat}{tag}_{X.shape[1]}"
    finish(name, X, y, f"GPT-2 OWT hidden state -> log unigram freq of next token, "
                       f"{'standardize, ' if args.standardize else ''}{'sphere, ' if args.sphere else ''}{args.feat} D={X.shape[1]}")


def build_msd(args):
    path = RAW / "msd" / "yearpredictionmsd.zip"
    with zipfile.ZipFile(path) as zf:
        inner = [m for m in zf.namelist() if m.lower().endswith((".txt", ".csv"))][0]
        data = np.loadtxt(io.TextIOWrapper(zf.open(inner)), delimiter=",", dtype=np.float32)
    y, Z = data[:, 0], standardize_cols(data[:, 1:])
    y = (y - y.mean()) / y.std()
    if args.sphere:
        Z = sphere(Z)
    X = features(Z, args.feat, args.D, args.seed)
    name = args.name or f"msd_year_{args.feat}{'s' if args.sphere else ''}_{X.shape[1]}"
    finish(name, X, y, f"YearPredictionMSD year (standardized), {args.feat} D={X.shape[1]}")


def build_higgs(args):
    path = RAW / "higgs" / "higgs.zip"
    with zipfile.ZipFile(path) as zf:
        inner = zf.namelist()[0]
        raw = zf.open(inner)
        if inner.endswith(".gz"):
            import gzip
            raw = gzip.open(raw)
        n = args.n or 11_000_000
        data = np.loadtxt(io.TextIOWrapper(raw), delimiter=",", dtype=np.float32,
                          max_rows=n)
    y, Z = 2 * data[:, 0] - 1, standardize_cols(data[:, 1:])
    X = features(Z, args.feat, args.D, args.seed)
    name = args.name or f"higgs_{args.feat}_{X.shape[1]}"
    finish(name, X, y, f"HIGGS +-1, {args.feat} D={X.shape[1]}, n={len(y)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", choices=["cifar", "owt", "msd", "higgs"])
    ap.add_argument("--feat", default="sketch",
                    help="raw | sketch | relu | tanh | relu+sketch | tanh+sketch")
    ap.add_argument("--D", type=int, default=2048)
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--target", default="animal")
    ap.add_argument("--sphere", action="store_true")
    ap.add_argument("--standardize", action="store_true")
    ap.add_argument("--name", default="")
    args = ap.parse_args()
    {"cifar": build_cifar, "owt": build_owt, "msd": build_msd,
     "higgs": build_higgs}[args.dataset](args)


if __name__ == "__main__":
    main()
