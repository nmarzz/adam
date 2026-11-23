"""Submit the slurm jobs of the full-training frequency-kink experiment.

Plan: ``experiments/Frequency kink in full GPT-2 training.md``. Every job runs through
``results/gptkink/job.sbatch`` (one H100, 23h50 limit, checkpoint on USR1). Run from the
repository root; add --dry-run to print the sbatch commands without submitting.

    lm/bin/python experiments/gptkink_launch.py prep      # tokenize OWT (once, ~1-2 h)
    lm/bin/python experiments/gptkink_launch.py smoke     # Stage 0 checks (~30 min)
    lm/bin/python experiments/gptkink_launch.py tune      # lr grid, 10% budget (20 short jobs)
    lm/bin/python experiments/gptkink_launch.py status    # tuning table, runs, probes
    lm/bin/python experiments/gptkink_launch.py stage1    # 8 runs + dependent probe jobs
    lm/bin/python experiments/gptkink_launch.py resume    # resubmit unfinished runs / probes
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results" / "gptkink"
CK = ROOT / "data" / "gptkink" / "ckpt"
PY = "lm/bin/python"
JOB = "results/gptkink/job.sbatch"

L = 256
BS = {1024: 4, 4096: 16, 16384: 64, 65536: 256}     # b (tokens/step) -> B (sequences)
LR0, B0 = 1e-3, 16384                              # sqrt-rule anchor
GRID = [0.25, 0.5, 1.0, 2.0, 4.0]
HEAD_BETA2 = [0.9, 0.99]
TUNE_TOKENS, RUN_TOKENS = 1e8, 1e9


def name(b, beta2):
    return f"b{b // 1024}k_h{round(beta2 * 100)}"


def sbatch(job, cmd, dry, extra=(), dep=None):
    args = ["sbatch", "--parsable", "--job-name", job, *extra]
    if dep:
        args.append(f"--dependency=afterany:{dep}")
    args += [JOB, *cmd]
    print(" ".join(args))
    if dry:
        return None
    r = subprocess.run(args, cwd=ROOT, capture_output=True, text=True, check=True)
    jid = r.stdout.strip().split(";")[0]
    print(f"  -> job {jid}")
    return jid


def train_cmd(run, b, lr, beta2, tokens, tune=False):
    c = [PY, "experiments/gptkink_train.py", "--name", run, "--B", str(BS[b]), "--lr", f"{lr:.3g}",
         "--head-beta2", str(beta2), "--tokens", f"{tokens:g}"]
    return c + (["--tune"] if tune else [])


def tuned_lrs():
    best = {}
    for f in sorted((RES / "tune").glob("*.json")):
        d = json.loads(f.read_text())
        if d.get("final_loss") is None:
            continue
        cur = best.get(d["b"])
        if cur is None or d["final_loss"] < cur["final_loss"]:
            best[d["b"]] = d
    return best


def running_jobs():
    r = subprocess.run(["squeue", "--me", "-h", "-o", "%j"], capture_output=True, text=True)
    return set(r.stdout.split())


def cmd_prep(a):
    sbatch("gk_prep", [PY, "experiments/gptkink_prep.py", "--shards", "20", "--procs", "16"], a.dry_run,
           extra=["-c", "16", "--mem", "64G", "-t", "0-08:00"])


def cmd_smoke(a):
    run = "smoke_b16k"
    cmd = ["bash", "-c",
           f"{PY} experiments/gptkink_train.py --name {run} --B 64 --lr 1e-3 --tokens 5e7 --warmup 200 "
           f"--probe-fracs 0,1 --out results/gptkink/smoke && "
           f"{PY} experiments/gptkink_probe.py --run {run} --tokens 1e7 --out results/gptkink/smoke"]
    sbatch("gk_smoke", cmd, a.dry_run, extra=["-t", "0-02:00"])


def cmd_tune(a):
    for b in BS:
        for g in GRID:
            lr = LR0 * (b / B0) ** 0.5 * g
            run = f"tune_b{b // 1024}k_lr{lr:.3g}"
            if (RES / "tune" / f"{run}.json").exists():
                continue
            sbatch(f"gk_{run}", train_cmd(run, b, lr, 0.99, TUNE_TOKENS, tune=True), a.dry_run,
                   extra=["-t", "0-06:00"])


def cmd_status(a):
    print("== tuning (final held-out loss after 1e8 tokens)")
    rows = {}
    for f in sorted((RES / "tune").glob("*.json")):
        d = json.loads(f.read_text())
        rows.setdefault(d["b"], []).append((d["lr"], d.get("final_loss")))
    for b in sorted(rows):
        r = sorted(rows[b])
        fin = [x for x in r if x[1] is not None]
        best = min(fin, key=lambda x: x[1]) if fin else None
        edge = best is not None and best[0] in (r[0][0], r[-1][0])
        print(f"  b={b:>6}: " + "  ".join(f"{lr:.3g}:{'div' if l is None else f'{l:.3f}'}" for lr, l in r)
              + (f"   best {best[0]:.3g}" if best else "") + ("  (AT GRID EDGE)" if edge else ""))
    print("== runs")
    for b in BS:
        for beta2 in HEAD_BETA2:
            run = name(b, beta2)
            st = "done" if (CK / run / "DONE").exists() else ("partial" if (CK / run / "last.pt").exists()
                                                             else ("started" if (CK / run).exists() else "-"))
            probes = sorted(p.parent.parent.name[5:] for p in (RES / "probes").glob(f"frac_*/runs/{run}.npz"))
            print(f"  {run:10s} {st:8s} probes: {','.join(probes) or '-'}")


def cmd_stage1(a, only_missing=False):
    best = tuned_lrs()
    missing = [b for b in BS if b not in best]
    if missing and not a.lr_from_rule:
        raise SystemExit(f"no tuning result for b = {missing}; run `tune` first (or pass --lr-from-rule)")
    queued = running_jobs()
    for b in BS:
        lr = LR0 * (b / B0) ** 0.5 if a.lr_from_rule else best[b]["lr"]
        for beta2 in HEAD_BETA2:
            run = name(b, beta2)
            if f"gk_{run}" in queued or f"gk_probe_{run}" in queued:
                print(f"  {run}: already queued/running, skipping")
                continue
            if (CK / run / "DONE").exists():
                jid = None
            else:
                jid = sbatch(f"gk_{run}", train_cmd(run, b, lr, beta2, RUN_TOKENS), a.dry_run)
            n_ck = len(list((CK / run).glob("probe_*.pt"))) if (CK / run).exists() else 0
            n_pr = len(list((RES / "probes").glob(f"frac_*/runs/{run}.npz")))
            if not only_missing or jid or n_pr < max(n_ck, 7):
                sbatch(f"gk_probe_{run}", [PY, "experiments/gptkink_probe.py", "--run", run], a.dry_run,
                       extra=["-t", "0-06:00"], dep=jid)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["prep", "smoke", "tune", "status", "stage1", "resume"])
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--lr-from-rule", action="store_true", help="stage1 without tuning: sqrt rule")
    a = ap.parse_args()
    (RES / "slurm").mkdir(parents=True, exist_ok=True)
    if a.cmd == "resume":
        cmd_stage1(a, only_missing=True)
    else:
        {"prep": cmd_prep, "smoke": cmd_smoke, "tune": cmd_tune, "status": cmd_status,
         "stage1": cmd_stage1}[a.cmd](a)


if __name__ == "__main__":
    main()
