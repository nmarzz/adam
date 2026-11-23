"""Tokenize OpenWebText for the full-training frequency-kink experiment.

Plan: ``experiments/Frequency kink in full GPT-2 training.md``.

Writes to ``--out`` (default ``data/gptkink/``):

* ``train.bin``          uint16 token stream (documents joined with <|endoftext|>),
* ``test.bin``           uint16 held-out stream: the same documents, in the same order,
                         as the held-out set of ``data/freqkink`` (head-only runs),
* ``train_counts.npy``   (50304,) int64 token counts of train.bin (padding ids are 0),
* ``rho_L{L}.npy``       (50304,) fraction of random (L+1)-token windows whose targets
                         contain each token (burstiness-aware event frequency),
* ``meta.json``.

The held-out documents are positions [35031, 38481) of the seed-0 permutation of the
documents in OWT shards 0-9 (``data/freqkink/meta.json``); they are removed from the
training stream. Training documents are all other documents of shards 0..S-1, in
shard order.

Usage (from the repository root, torch venv):
    lm/bin/python experiments/gptkink_prep.py --shards 20 --procs 16
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OWT = Path.home() / ".cache/huggingface/datasets/openwebtext/plain_text/1.0.0"
VOCAB, VPAD, EOS = 50257, 50304, 50256
HELD = (35031, 38481)  # positions in the seed-0 permutation of shards 0-9 (freqkink_prep)


def shard_files():
    return sorted(glob.glob(str(OWT / "*" / "*.arrow")))


def _tok():
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    from transformers import GPT2TokenizerFast
    t = GPT2TokenizerFast.from_pretrained("gpt2")
    t.model_max_length = 10**9
    return t


def tokenize_shard(job):
    """Tokenize one shard (skipping excluded local indices) into a uint16 file."""
    path, excl, out, max_docs = job
    from datasets import Dataset
    ds = Dataset.from_file(path)
    n = len(ds) if max_docs is None else min(len(ds), max_docs)
    keep = np.setdiff1d(np.arange(n), np.asarray(excl, dtype=np.int64))
    tok = _tok()
    total = 0
    with open(out, "wb") as f:
        for a in range(0, len(keep), 1000):
            texts = ds[keep[a:a + 1000].tolist()]["text"]
            ids = tok(texts, add_special_tokens=False)["input_ids"]
            arr = np.concatenate([np.asarray(x + [EOS], np.uint16) for x in ids])
            arr.tofile(f)
            total += len(arr)
    return out, total, len(keep)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", type=int, default=20)
    ap.add_argument("--procs", type=int, default=8)
    ap.add_argument("--out", default=str(ROOT / "data" / "gptkink"))
    ap.add_argument("--max-docs", type=int, default=None, help="per shard (smoke tests)")
    ap.add_argument("--L", type=int, default=256)
    ap.add_argument("--rho-windows", type=int, default=2_000_000)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    from datasets import Dataset

    files = shard_files()
    assert len(files) >= max(args.shards, 10), "need the cached OWT shards"
    lens = [len(Dataset.from_file(f)) for f in files[:10]]
    offs = np.concatenate([[0], np.cumsum(lens)])
    order = np.random.default_rng(0).permutation(offs[-1])
    held = order[HELD[0]:HELD[1]]
    print(f"{offs[-1]:,} documents in shards 0-9; {len(held):,} held-out documents", flush=True)

    # ---- held-out stream: same documents and order as data/freqkink ----
    tok = _tok()
    dss = [Dataset.from_file(f) for f in files[:10]]
    test = []
    for g in held:
        s = int(np.searchsorted(offs, g, side="right") - 1)
        ids = tok(dss[s][int(g - offs[s])]["text"], add_special_tokens=False)["input_ids"]
        test.append(np.asarray(ids + [EOS], np.uint16))
    test = np.concatenate(test)
    test.tofile(out / "test.bin")
    print(f"test.bin: {len(test):,} tokens ({time.time() - t0:.0f}s)", flush=True)

    # ---- training stream ----
    jobs = []
    for s in range(args.shards):
        excl = (held[(held >= offs[s]) & (held < offs[s + 1])] - offs[s]).tolist() if s < 10 else []
        jobs.append((files[s], excl, str(out / f"train_{s:02d}.bin"), args.max_docs))
    with Pool(args.procs) as pool:
        parts = []
        for path, n, nd in pool.imap(tokenize_shard, jobs):
            parts.append((path, n))
            print(f"  {Path(path).name}: {n:,} tokens from {nd:,} documents ({time.time() - t0:.0f}s)", flush=True)
    total = sum(n for _, n in parts)
    train = np.memmap(out / "train.bin", np.uint16, "w+", shape=(total,))
    a = 0
    for path, n in parts:
        train[a:a + n] = np.fromfile(path, np.uint16)
        a += n
        os.remove(path)
    train.flush()
    print(f"train.bin: {total:,} tokens ({time.time() - t0:.0f}s)", flush=True)

    # ---- counts and burstiness-aware window frequencies ----
    counts = np.zeros(VPAD, np.int64)
    for a in range(0, total, 1 << 28):
        counts += np.bincount(train[a:a + (1 << 28)], minlength=VPAD)
    np.save(out / "train_counts.npy", counts)
    rng = np.random.default_rng(1)
    L, nw = args.L, min(args.rho_windows, max(total // (args.L + 1), 1))
    pres = np.zeros(VPAD, np.int64)
    for a in range(0, nw, 100_000):
        m = min(100_000, nw - a)
        st = rng.integers(0, total - L - 1, m)
        w = np.sort(train[st[:, None] + 1 + np.arange(L)], axis=1)  # targets of each window
        first = np.ones_like(w, dtype=bool)
        first[:, 1:] = w[:, 1:] != w[:, :-1]
        pres += np.bincount(w[first], minlength=VPAD)
    np.save(out / f"rho_L{L}.npy", pres / nw)
    meta = dict(shards=args.shards, max_docs=args.max_docs, train_tokens=int(total),
                test_tokens=int(len(test)), held_positions=HELD, vocab=VOCAB, vocab_padded=VPAD,
                eos=EOS, rho_L=L, rho_windows=int(nw), never_seen=int((counts[:VOCAB] == 0).sum()))
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta), flush=True)


if __name__ == "__main__":
    main()
