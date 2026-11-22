"""GPT-2 small final hidden states on OpenWebText for the frequency-kink experiment.

Writes to ``data/freqkink/``:

* ``{split}_X.npy``  (n, 768) float16, GPT-2 ``ln_f`` output at each kept position,
* ``{split}_y.npy``  (n,) int32, the next token (target),
* ``meta.json``      settings and sizes.

Documents are drawn in a random order from the cached OpenWebText shards; the
first ``--n-train`` targets come from one set of documents and ``--n-test``
targets from the following, disjoint documents. Documents are joined with
``<|endoftext|>`` and run through GPT-2 in windows of 1024 tokens with stride
512; only the last 512 positions of each window are kept (the first window keeps
all), so every target is predicted from at least 512 tokens of context and each
token of the stream is a target exactly once.

Usage (from the repository root, in the torch venv):
    lm/bin/python experiments/freqkink_prep.py --n-train 40000000 --n-test 4000000
"""
from __future__ import annotations

import argparse
import glob
import json
import time
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset, concatenate_datasets
from transformers import GPT2Model, GPT2TokenizerFast

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "freqkink"
OWT = Path.home() / ".cache/huggingface/datasets/openwebtext/plain_text/1.0.0"
CTX, STRIDE = 1024, 512


def token_stream(ds, order, tok, start):
    """Yield (doc_position, token ids) for documents ds[order[start:]]."""
    eos = tok.eos_token_id
    batch = 256
    for a in range(start, len(order), batch):
        idx = order[a:a + batch].tolist()
        texts = ds[idx]["text"]
        enc = tok(texts, add_special_tokens=False)["input_ids"]
        for k, ids in enumerate(enc):
            yield a + k + 1, np.asarray(ids + [eos], np.int32)


def windows(stream, need):
    """Collect ``need`` + context tokens and cut them into stride-512 windows.

    Returns (inputs (W, 1024), keep_from (W,), next document position).
    Window w covers tokens [w*512, w*512+1024] (inputs and shifted targets).
    """
    buf, total, pos = [], 0, 0
    goal = need + STRIDE + 1
    for pos, ids in stream:
        buf.append(ids)
        total += len(ids)
        if total >= goal:
            break
    toks = np.concatenate(buf)
    n_win = (need - (CTX - STRIDE)) // STRIDE + 1
    n_win = max(n_win, 1)
    while (n_win - 1) * STRIDE + CTX + 1 > len(toks):
        n_win -= 1
    starts = np.arange(n_win) * STRIDE
    return toks, starts, pos


@torch.no_grad()
def embed_split(name, model, toks, starts, n_keep, dev, bs):
    """Run windows through GPT-2 and stream kept (X, y) to disk."""
    X = np.lib.format.open_memmap(OUT / f"{name}_X.npy", "w+", np.float16, (n_keep, 768))
    Y = np.lib.format.open_memmap(OUT / f"{name}_y.npy", "w+", np.int32, (n_keep,))
    ptr, t0 = 0, time.time()
    for a in range(0, len(starts), bs):
        s = starts[a:a + bs]
        inp = np.stack([toks[i:i + CTX] for i in s])
        tgt = np.stack([toks[i + 1:i + CTX + 1] for i in s])
        h = model(torch.from_numpy(inp).long().to(dev)).last_hidden_state
        h = h.float().cpu().numpy()
        for j, st in enumerate(s):
            lo = 0 if st == 0 else CTX - STRIDE
            k = min(CTX - lo, n_keep - ptr)
            if k <= 0:
                break
            X[ptr:ptr + k] = h[j, lo:lo + k]
            Y[ptr:ptr + k] = tgt[j, lo:lo + k]
            ptr += k
        if (a // bs) % 50 == 0:
            rate = ptr / (time.time() - t0)
            print(f"  {name}: {ptr:,}/{n_keep:,} positions  ({rate:,.0f}/s)", flush=True)
        if ptr >= n_keep:
            break
    X.flush(); Y.flush()
    if ptr < n_keep:
        raise RuntimeError(f"{name}: only {ptr} positions produced")
    return ptr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=40_000_000)
    ap.add_argument("--n-test", type=int, default=4_000_000)
    ap.add_argument("--shards", type=int, default=10, help="OWT shards to draw documents from")
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    files = sorted(glob.glob(str(OWT / "*" / "*.arrow")))[: args.shards]
    ds = concatenate_datasets([Dataset.from_file(f) for f in files])
    order = np.random.default_rng(args.seed).permutation(len(ds))
    print(f"{len(ds):,} documents from {len(files)} shards")

    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    tok.model_max_length = 10**9
    dev = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True
    model = GPT2Model.from_pretrained("gpt2").to(dev).eval()

    meta = {"model": "gpt2", "layer": "ln_f (last_hidden_state)", "ctx": CTX, "stride": STRIDE,
            "shards": len(files), "seed": args.seed}
    start = 0
    for name, n in (("train", args.n_train), ("test", args.n_test)):
        t0 = time.time()
        toks, starts, start = windows(token_stream(ds, order, tok, start), n)
        print(f"{name}: {len(toks):,} tokens from documents up to #{start:,} "
              f"({time.time() - t0:.0f}s tokenizing), {len(starts):,} windows")
        got = embed_split(name, model, toks, starts, n, dev, args.bs)
        meta[name] = {"n": got, "docs_end": start}
        print(f"{name}: done in {time.time() - t0:.0f}s")
    (OUT / "meta.json").write_text(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
