"""Full GPT-2-style training for the frequency-kink experiment.

Plan: ``experiments/Frequency kink in full GPT-2 training.md``. Data from
``experiments/gptkink_prep.py``.

Model: pre-LN GPT-2 block stack (6 layers, d = 384, 6 heads, context L = 256 by
default), learned positions, vocabulary padded to 50304. The output head is a
separate (untied) matrix initialised at zero, kept in float32 with its own Adam
hyper-parameters (beta1, beta2, eps; no weight decay). Every head column without an
own event then receives exactly the same update as the never-seen padding columns,
so the own-event gain

    G_c = (w_c - w_u) . sum_{held-out i: y_i = c} h_i / n_c     (u = padding id 50257)

is exact (computed elementwise in float32), and l_c = (lse - z_u) - G_c.

Training: one optimizer step per b = B * L target tokens, windows of L + 1 tokens drawn
i.i.d. from the training stream (seeded by step, so runs resume deterministically),
warmup-stable-decay schedule (linear warmup over --warmup steps, constant, linear decay
to 0 over the last --decay-frac of the steps), global-norm clipping. The trunk uses AdamW
(beta = 0.9, 0.95, weight decay 0.1 on matrices and embeddings).

Evaluation (log-spaced steps plus every 5% of the run) on the held-out stream, windows of
L tokens with stride L/2 (every target has >= L/2 tokens of context): per-token sums of
loss, own logit and padding-column logit, the exact gain, logits on the mean feature,
head sqrt(v_hat) statistics and the shared-mean fraction of the features. Records are
written to ``results/gptkink/runs/<name>.npz`` in the schema of
``freqkink_analysis.py``. Checkpoints for the local frozen-drift probes
(``gptkink_probe.py``) are written at --probe-fracs of the run.

Resumable: re-running the same command continues from ``<ckpt>/<name>/last.pt``.

Usage (from the repository root, torch venv):
    lm/bin/python experiments/gptkink_train.py --name b16k_h99 --B 64 --lr 1e-3 --head-beta2 0.99
"""
from __future__ import annotations

import argparse
import json
import math
import os
import signal
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
VOCAB, VPAD, EOS = 50257, 50304, 50256
U = VOCAB  # first padding id: a token that never occurs (background column)


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
class Attention(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.h = h
        self.c_attn = nn.Linear(d, 3 * d)
        self.c_proj = nn.Linear(d, d)

    def forward(self, x):
        B, T, d = x.shape
        q, k, v = self.c_attn(x).split(d, dim=2)
        q, k, v = (t.view(B, T, self.h, d // self.h).transpose(1, 2) for t in (q, k, v))
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.c_proj(y.transpose(1, 2).contiguous().view(B, T, d))


class Block(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.ln_1 = nn.LayerNorm(d)
        self.attn = Attention(d, h)
        self.ln_2 = nn.LayerNorm(d)
        self.c_fc = nn.Linear(d, 4 * d)
        self.c_proj = nn.Linear(4 * d, d)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        return x + self.c_proj(F.gelu(self.c_fc(self.ln_2(x)), approximate="tanh"))


class GPT(nn.Module):
    def __init__(self, n_layer=6, d=384, n_head=6, L=256):
        super().__init__()
        self.wte = nn.Embedding(VPAD, d)
        self.wpe = nn.Embedding(L, d)
        self.blocks = nn.ModuleList([Block(d, n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(d)
        self.head = nn.Parameter(torch.zeros(VPAD, d))  # untied, zero init, float32
        for name, p in self.named_parameters():
            if name == "head":
                continue
            if p.dim() >= 2:
                std = 0.02 / math.sqrt(2 * n_layer) if name.endswith("c_proj.weight") else 0.02
                nn.init.normal_(p, 0.0, std)
            elif name.endswith("bias"):
                nn.init.zeros_(p)

    def features(self, idx):
        T = idx.shape[1]
        x = self.wte(idx) + self.wpe(torch.arange(T, device=idx.device))
        for blk in self.blocks:
            x = blk(x)
        return self.ln_f(x)

    def logits(self, h):
        with torch.autocast("cuda", enabled=False):
            return h.float() @ self.head.t()


def make_optimizer(model, a):
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if name == "head":
            continue
        (decay if p.dim() >= 2 else no_decay).append(p)
    groups = [
        dict(params=decay, weight_decay=a.trunk_wd, betas=tuple(a.trunk_betas), eps=1e-8, name="trunk"),
        dict(params=no_decay, weight_decay=0.0, betas=tuple(a.trunk_betas), eps=1e-8, name="trunk_nd"),
        dict(params=[model.head], weight_decay=0.0, betas=(a.head_beta1, a.head_beta2), eps=a.head_eps,
             name="head"),
    ]
    return torch.optim.AdamW(groups, lr=a.lr, fused=True)


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
class Data:
    def __init__(self, path, L, n_test, device):
        self.train = np.memmap(Path(path) / "train.bin", np.uint16, "r")
        self.L = L
        self.train_counts = np.load(Path(path) / "train_counts.npy")
        test = np.fromfile(Path(path) / "test.bin", np.uint16).astype(np.int64)
        stride = L // 2
        n_win = (len(test) - L - 1) // stride + 1
        starts = np.arange(n_win) * stride
        keep_from = np.full(n_win, L - stride)
        keep_from[0] = 0
        kept = np.cumsum(L - keep_from)
        n_win = int(np.searchsorted(kept, n_test) + 1) if kept[-1] >= n_test else n_win
        self.test = torch.from_numpy(test).to(device)
        self.starts = torch.from_numpy(starts[:n_win]).to(device)
        self.keep_from = torch.from_numpy(keep_from[:n_win]).to(device)
        self.n_test = min(int(n_test), int(kept[n_win - 1]))
        self.device = device

    def batch(self, B, step, seed):
        rng = np.random.default_rng([seed, step])
        st = rng.integers(0, len(self.train) - self.L - 1, B)
        w = torch.from_numpy(np.stack([self.train[s:s + self.L + 1] for s in st]).astype(np.int64))
        w = w.pin_memory().to(self.device, non_blocking=True)
        return w[:, :-1], w[:, 1:]


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #
@torch.no_grad()
def held_out_pass(model, data, wins=64, with_logits=True):
    """Per-token sums over the held-out targets. Returns a dict of CPU numpy arrays."""
    dev, L, d = data.device, data.L, model.head.shape[1]
    ar = torch.arange(L, device=dev)
    S = {k: torch.zeros(VPAD, device=dev) for k in ("loss", "own", "bg", "n")}
    ctx = torch.zeros(VPAD, d, device=dev)
    hsum, hsq, seen = torch.zeros(d, device=dev), torch.zeros((), device=dev), 0
    for a in range(0, len(data.starts), wins):
        st = data.starts[a:a + wins]
        x = data.test[st[:, None] + ar]
        y = data.test[st[:, None] + 1 + ar]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            h = model.features(x)
        keep = ar[None, :] >= data.keep_from[a:a + wins, None]
        hk, yk = h.float()[keep], y[keep]
        if seen + len(yk) > data.n_test:
            hk, yk = hk[:data.n_test - seen], yk[:data.n_test - seen]
        seen += len(yk)
        ctx.index_add_(0, yk, hk)
        S["n"].index_add_(0, yk, torch.ones_like(yk, dtype=torch.float32))
        hsum += hk.sum(0)
        hsq += (hk * hk).sum()
        if with_logits:
            z = model.logits(hk)
            lse = torch.logsumexp(z, 1)
            own = z.gather(1, yk[:, None])[:, 0]
            S["loss"].index_add_(0, yk, lse - own)
            S["own"].index_add_(0, yk, own)
            S["bg"].index_add_(0, yk, z[:, U])
        if seen >= data.n_test:
            break
    W = model.head.float()
    hbar = hsum / seen
    out = {k: v.cpu().numpy() for k, v in S.items()}
    out["gain"] = ((W - W[U]) * ctx).sum(1).cpu().numpy()
    out["mean_ctx_logit"] = (W @ hbar).cpu().numpy()
    out["shared_mean_frac"] = float((hbar @ hbar) / (hsq / seen))
    out["ctx"] = ctx
    return out


def head_v_stats(opt, model, beta2):
    st = opt.state.get(model.head, {})
    if "exp_avg_sq" not in st:
        return None, None
    step = float(st["step"])
    vhat = st["exp_avg_sq"] / (1 - beta2 ** step)
    return (vhat.mean(1).sqrt().cpu().numpy().astype(np.float32),
            vhat.sqrt().median(1).values.cpu().numpy().astype(np.float32))


def pad_spread(model):
    P = model.head[VOCAB:]
    return float((P - P[:1]).abs().max()), float(P[:1].abs().max())


# --------------------------------------------------------------------------- #
# Schedule
# --------------------------------------------------------------------------- #
def lr_factor(t, T, warmup, decay_frac):
    f = min(1.0, (t + 1) / warmup)
    t_dec = int(T * (1 - decay_frac))
    if t >= t_dec:
        f *= max(0.0, (T - t) / max(T - t_dec, 1))
    return f


def eval_steps(T, per_octave, every_frac):
    s = set(np.unique(np.round(np.geomspace(1, T, int(per_octave * np.log2(max(T, 2))) + 1)).astype(int)))
    s |= {int(round(T * k * every_frac)) for k in range(1, int(round(1 / every_frac)) + 1)}
    s |= {0, T}
    return sorted(x for x in s if 0 <= x <= T)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--B", type=int, required=True, help="sequences per optimizer step (b = B * L)")
    ap.add_argument("--micro", type=int, default=64, help="sequences per micro-batch")
    ap.add_argument("--L", type=int, default=256)
    ap.add_argument("--n-layer", type=int, default=6)
    ap.add_argument("--d", type=int, default=384)
    ap.add_argument("--n-head", type=int, default=6)
    ap.add_argument("--tokens", type=float, default=1e9, help="training target tokens")
    ap.add_argument("--lr", type=float, required=True)
    ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument("--decay-frac", type=float, default=0.1)
    ap.add_argument("--head-beta1", type=float, default=0.0)
    ap.add_argument("--head-beta2", type=float, default=0.99)
    ap.add_argument("--head-eps", type=float, default=1e-8)
    ap.add_argument("--trunk-betas", type=float, nargs=2, default=[0.9, 0.95])
    ap.add_argument("--trunk-wd", type=float, default=0.1)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-test", type=float, default=4e6)
    ap.add_argument("--per-octave", type=int, default=4)
    ap.add_argument("--every-frac", type=float, default=0.05)
    ap.add_argument("--probe-fracs", type=str, default="0,0.01,0.03,0.1,0.3,0.6,1")
    ap.add_argument("--ckpt-minutes", type=float, default=30)
    ap.add_argument("--tune", action="store_true", help="evaluate only at the end; write tune/<name>.json")
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--data", default=str(ROOT / "data" / "gptkink"))
    ap.add_argument("--out", default=str(ROOT / "results" / "gptkink"))
    ap.add_argument("--ckpt", default=str(ROOT / "data" / "gptkink" / "ckpt"))
    return ap.parse_args()


def main():
    a = parse()
    a.data, a.out, a.ckpt = (str(Path(x).resolve()) for x in (a.data, a.out, a.ckpt))
    torch.backends.cuda.matmul.allow_tf32 = False  # keep the float32 head exact
    torch.backends.cudnn.allow_tf32 = False
    dev = torch.device("cuda")
    b = a.B * a.L
    T = int(a.tokens // b)
    out, ck = Path(a.out), Path(a.ckpt) / a.name
    (out / ("tune" if a.tune else "runs")).mkdir(parents=True, exist_ok=True)
    ck.mkdir(parents=True, exist_ok=True)
    done_file = out / ("tune" if a.tune else "runs") / f"{a.name}.{'json' if a.tune else 'npz'}"
    if (ck / "DONE").exists():
        print(f"{a.name}: already finished ({done_file})", flush=True)
        return

    torch.manual_seed(a.seed)
    data = Data(a.data, a.L, a.n_test, dev)
    model = GPT(a.n_layer, a.d, a.n_head, a.L).to(dev)
    opt = make_optimizer(model, a)
    fwd = torch.compile(model.features) if a.compile else model.features
    cfg = dict(vars(a), b=b, T=T, opt="adam", beta2=a.head_beta2, eps=a.head_eps, frozen=False, center=False)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"{a.name}: b={b:,} tokens/step, T={T:,} steps, {n_params / 1e6:.1f}M params "
          f"(head {model.head.numel() / 1e6:.1f}M), n_test={data.n_test:,}", flush=True)

    rec, step, tau, n_clip, t_used = [], 0, 0.0, 0, 0.0
    if (ck / "last.pt").exists():
        s = torch.load(ck / "last.pt", map_location=dev, weights_only=False)
        model.load_state_dict(s["model"]); opt.load_state_dict(s["opt"])
        rec, step, tau, n_clip, t_used = s["rec"], s["step"], s["tau"], s["n_clip"], s["t_used"]
        print(f"resumed at step {step:,}", flush=True)

    ev = [T] if a.tune else eval_steps(T, a.per_octave, a.every_frac)
    probe_steps = {} if a.tune else {int(round(float(f) * T)): float(f) for f in a.probe_fracs.split(",")}
    stop = {"flag": False}
    signal.signal(signal.SIGTERM, lambda *_: stop.update(flag=True))
    signal.signal(signal.SIGUSR1, lambda *_: stop.update(flag=True))

    def save_last():
        torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), rec=rec, step=step, tau=tau,
                        n_clip=n_clip, t_used=t_used + time.time() - t0), ck / "last.pt.tmp")
        os.replace(ck / "last.pt.tmp", ck / "last.pt")

    def evaluate():
        model.eval()
        r = held_out_pass(model, data)
        model.train()
        vr, vm = head_v_stats(opt, model, a.head_beta2)
        sp, sc = pad_spread(model)
        r.pop("ctx")
        r.update(step=step, tau=tau, v_rms=vr, v_med=vm, pad_spread=sp, pad_scale=sc,
                 clip_frac=n_clip / max(step, 1), wall=t_used + time.time() - t0)
        rec.append(r)
        tot = r["loss"].sum() / data.n_test
        print(f"  step {step:>9,} ({step * b:>13,} tok)  loss {tot:.4f}  G[pad spread {sp:.1e}]  "
              f"shared-mean {r['shared_mean_frac']:.3f}  clip {r['clip_frac']:.3f}  [{r['wall'] / 60:.1f} min]",
              flush=True)
        return tot

    def write_records():
        if a.tune:
            tot = rec[-1]["loss"].sum() / data.n_test
            done_file.write_text(json.dumps(dict(cfg, final_loss=float(tot) if np.isfinite(tot) else None)))
            return
        arr = lambda k: np.stack([r[k] for r in rec]).astype(np.float32)
        o = dict(steps=np.array([r["step"] for r in rec]), samples=np.array([r["step"] * b for r in rec]),
                 tau=np.array([r["tau"] for r in rec]), per_tok_sum=arr("loss"), own_logit_sum=arr("own"),
                 bg_logit_sum=arr("bg"), gain_sum=arr("gain"), mean_ctx_logit=arr("mean_ctx_logit"),
                 total=np.array([r["loss"].sum() / data.n_test for r in rec]),
                 shared_mean_frac=np.array([r["shared_mean_frac"] for r in rec]),
                 pad_spread=np.array([[r["pad_spread"], r["pad_scale"]] for r in rec]),
                 clip_frac=np.array([r["clip_frac"] for r in rec]), wall=np.array([r["wall"] for r in rec]))
        vs = [r for r in rec if r["v_rms"] is not None]
        if vs:
            o["v_steps"] = np.array([r["step"] for r in vs])
            o["v_rms"] = np.stack([r["v_rms"] for r in vs]); o["v_med"] = np.stack([r["v_med"] for r in vs])
        np.savez_compressed(done_file, **o, train_counts=data.train_counts, test_counts=rec[0]["n"].astype(np.int64),
                            n_test=data.n_test, cfg=json.dumps(cfg))

    def save_probe(frac):
        torch.save(dict(model=model.state_dict(), head_state=opt.state.get(model.head, {}), step=step,
                        frac=frac, cfg=cfg), ck / f"probe_{frac:g}.pt")

    t0, last_ck = time.time(), time.time()
    n_micro = math.ceil(a.B / a.micro)
    model.train()
    while True:
        if step in ev and (not rec or rec[-1]["step"] != step):
            tot = evaluate()
            write_records()
            if not np.isfinite(tot):
                print("diverged", flush=True)
                break
        if step in probe_steps and not (ck / f"probe_{probe_steps[step]:g}.pt").exists():
            save_probe(probe_steps[step])
        if step >= T:
            break
        if stop["flag"] or time.time() - last_ck > 60 * a.ckpt_minutes:
            save_last()
            last_ck = time.time()
            if stop["flag"]:
                print(f"signal: checkpointed at step {step:,}; resubmit to continue", flush=True)
                return
        # ---- one optimizer step over b target tokens ----
        f = lr_factor(step, T, a.warmup, a.decay_frac)
        for g in opt.param_groups:
            g["lr"] = a.lr * f
        x, y = data.batch(a.B, step, a.seed)
        for m in range(n_micro):
            xs, ys = x[m * a.micro:(m + 1) * a.micro], y[m * a.micro:(m + 1) * a.micro]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                h = fwd(xs)
            z = model.logits(h.reshape(-1, h.shape[-1]))
            loss = F.cross_entropy(z, ys.reshape(-1), reduction="sum") / b
            loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), a.clip)
        n_clip += int(gn.item() > a.clip)
        opt.step()
        opt.zero_grad(set_to_none=True)
        tau += a.lr * f
        step += 1

    if step >= T or not np.isfinite(rec[-1]["loss"].sum()):
        (ck / "DONE").write_text(time.strftime("%c"))
        (ck / "last.pt").unlink(missing_ok=True)
        print(f"{a.name}: finished -> {done_file}", flush=True)


if __name__ == "__main__":
    main()
