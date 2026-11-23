"""Local frozen-drift probe at a checkpoint of a gptkink_train.py run.

The trunk is frozen at the checkpoint; the head keeps its weights and its Adam state
(exp_avg, exp_avg_sq, step) and continues with lr = 1e-10 (default) on fresh training
windows, with the run's b, beta1, beta2 and eps. The denominators are then those of the
checkpoint (stationary, since W barely moves), so the own-event gain G grows at the
per-token drift given the current features: the head-only frozen-drift experiment
(freqkink.py) run on learned features.

At a trained checkpoint the net drift dG/dt is own-event forcing minus the restoring
pull of the background gradients (near a fixed point they cancel). The probe therefore
splits every head update exactly: for a given denominator the Adam step is linear in the
gradient, and the head gradient is the sum of own-event rows ((p_y - 1) h at each target)
and background rows (p_c h). ``gain_own_sum`` is the forcing part, ``gain_bg_sum`` the
background part, ``gain_sum`` their sum (the net drift). At W = 0 a token without own
events has no own part, so the own increment of the background column u is 0.

Because the trunk is frozen the held-out feature sums are fixed; they are computed
once. The checkpoint head W0 is held fixed and the probe updates are accumulated in a
separate tensor D (lr-sized steps would be lost in float32 if added to a trained W0), so
``gain_sum`` stores the increment (D_c - D_u) . sum h since the checkpoint (its slope is
the drift; ``gain0_sum`` is the gain at the checkpoint).

Output: ``results/gptkink/probes/frac_<f>/runs/<run>.npz`` in the schema read by
``freqkink_analysis.py`` (rate ``Gslope`` with ``--frozen``), so the probes of all runs
at one checkpoint fraction can be analysed together:

    FREQKINK_OUT=results/gptkink/probes/frac_0.1 FREQKINK_FIGS=figs/gptkink/frac_0.1 \
        real/bin/python experiments/freqkink_analysis.py --rate Gslope --frozen --norm plateau --pool

Usage (from the repository root, torch venv):
    lm/bin/python experiments/gptkink_probe.py --run b16k_h99            # every saved checkpoint
    lm/bin/python experiments/gptkink_probe.py --run b16k_h99 --fracs 0,0.1
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from gptkink_train import GPT, U, Data, held_out_pass

ROOT = Path(__file__).resolve().parents[1]


def probe(ck_path, a, dev):
    s = torch.load(ck_path, map_location=dev, weights_only=False)
    cfg, frac = s["cfg"], s["frac"]
    out_dir = Path(a.out) / "probes" / f"frac_{frac:g}" / "runs"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{cfg['name']}.npz"
    if out_file.exists() and not a.force:
        print(f"  {out_file} exists, skipping", flush=True)
        return
    t0 = time.time()
    model = GPT(cfg["n_layer"], cfg["d"], cfg["n_head"], cfg["L"]).to(dev)
    model.load_state_dict(s["model"])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    W0 = model.head.detach().clone().float()   # fixed: checkpoint head
    # Probe updates are accumulated apart from W0 (lr-sized steps are below the float32
    # resolution of a trained W0), split into the part driven by the own-event rows of the
    # head gradient and the part driven by the background rows (exact: for a given
    # denominator the Adam step is linear in the gradient).
    D_own, D_bg = torch.zeros_like(W0), torch.zeros_like(W0)
    hs = s["head_state"]
    m_bg = hs["exp_avg"].clone() if "exp_avg" in hs else torch.zeros_like(W0)  # checkpoint momentum
    m_own = torch.zeros_like(W0)                                               # (decays; beta1=0 here)
    v = hs["exp_avg_sq"].clone() if "exp_avg_sq" in hs else torch.zeros_like(W0)
    k = float(hs["step"]) if "step" in hs else 0.0
    b1, b2, eps = cfg["head_beta1"], cfg["head_beta2"], cfg["head_eps"]
    B, L = cfg["B"], cfg["L"]
    b = B * L
    data = Data(cfg["data"], L, cfg["n_test"], dev)

    ho = held_out_pass(model, data, with_logits=True)   # features fixed: sums computed once
    ctx, n = ho["ctx"], ho["n"]
    T = int(a.tokens // b)
    ev = set(np.unique(np.round(np.geomspace(1, T, int(a.per_octave * np.log2(max(T, 2))) + 1)).astype(int)))
    ev = sorted(ev | {0, T})
    g_own_rec, g_bg_rec, steps = [], [], []
    n_micro = math.ceil(B / cfg["micro"])
    seed = cfg["seed"] + 1_000_003  # windows disjoint in sequence from the training run's
    gain = lambda D: ((D - D[U]) * ctx).sum(1).cpu().numpy()
    for t in range(T + 1):
        if t in ev:  # increments of the gain since the checkpoint
            g_own_rec.append(gain(D_own)); g_bg_rec.append(gain(D_bg)); steps.append(t)
        if t == T:
            break
        x, y = data.batch(B, t, seed)
        g_all = torch.zeros_like(W0)
        g_own = torch.zeros_like(W0)
        with torch.no_grad():
            Wc = W0 + D_own + D_bg
            for j in range(n_micro):
                xs, ys = x[j * cfg["micro"]:(j + 1) * cfg["micro"]], y[j * cfg["micro"]:(j + 1) * cfg["micro"]]
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    h = model.features(xs)
                h = h.float().reshape(-1, h.shape[-1])
                ys = ys.reshape(-1)
                P = torch.softmax(h @ Wc.t(), dim=1)
                py = P.gather(1, ys[:, None])
                g_all += P.t() @ h
                g_all.index_add_(0, ys, -h)                  # dCE/dW = (P - Y)^T h
                g_own.index_add_(0, ys, -(1 - py) * h)       # own-event rows: (p_y - 1) h at the target
                del P
            g_all /= b
            g_own /= b
            g_bg = g_all - g_own                             # p_c h at every non-target column
            k += 1
            m_own.mul_(b1).add_(g_own, alpha=1 - b1)
            m_bg.mul_(b1).add_(g_bg, alpha=1 - b1)
            v.mul_(b2).addcmul_(g_all, g_all, value=1 - b2)
            denom = (v / (1 - b2 ** k)).sqrt_().add_(eps).mul_(1 - b1 ** k)
            D_own.addcdiv_(m_own, denom, value=-a.lr)
            D_bg.addcdiv_(m_bg, denom, value=-a.lr)
    pc = dict(cfg, lr=a.lr, frozen=True, probe_frac=frac, probe_tokens=a.tokens, run_lr=cfg["lr"])
    Go, Gb = np.stack(g_own_rec).astype(np.float32), np.stack(g_bg_rec).astype(np.float32)
    G = Go + Gb
    np.savez_compressed(out_file, steps=np.array(steps), samples=np.array(steps) * b, gain_sum=G,
                        gain_own_sum=Go, gain_bg_sum=Gb,
                        gain0_sum=ho["gain"].astype(np.float32),
                        per_tok_sum=np.repeat(ho["loss"][None], len(steps), 0).astype(np.float32),
                        own_logit_sum=np.repeat(ho["own"][None], len(steps), 0).astype(np.float32),
                        mean_ctx_logit=np.repeat(ho["mean_ctx_logit"][None], len(steps), 0).astype(np.float32),
                        total=np.full(len(steps), ho["loss"].sum() / data.n_test),
                        shared_mean_frac=ho["shared_mean_frac"],
                        train_counts=data.train_counts, test_counts=n.astype(np.int64),
                        n_test=data.n_test, cfg=json.dumps(pc))
    lin = lambda X: (lambda q: q[-1] / np.interp(T / 2, steps, q))(X[:, n > 0].sum(1) / n.sum())
    print(f"  {cfg['name']} frac {frac:g}: {T:,} probe steps, pooled dG(T)/dG(T/2): total {lin(G):.3f}, "
          f"own {lin(Go):.3f}; own/total at T {Go[-1, n > 0].sum() / G[-1, n > 0].sum():.3f}  "
          f"[{(time.time() - t0) / 60:.1f} min] -> {out_file}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--fracs", default=None, help="comma list; default: every saved checkpoint")
    ap.add_argument("--tokens", type=float, default=5e7)
    ap.add_argument("--lr", type=float, default=1e-10)
    ap.add_argument("--per-octave", type=int, default=4)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--out", default=str(ROOT / "results" / "gptkink"))
    ap.add_argument("--ckpt", default=str(ROOT / "data" / "gptkink" / "ckpt"))
    a = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    dev = torch.device("cuda")
    cks = sorted(Path(a.ckpt, a.run).glob("probe_*.pt"), key=lambda p: float(p.stem[6:]))
    if a.fracs is not None:
        want = {float(f) for f in a.fracs.split(",")}
        cks = [c for c in cks if float(c.stem[6:]) in want]
    if not cks:
        raise SystemExit(f"no probe checkpoints for {a.run}")
    for c in cks:
        probe(c, a, dev)


if __name__ == "__main__":
    main()
