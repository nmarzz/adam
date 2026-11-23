"""Submit the slurm jobs of the full-training frequency-kink experiment.

Plan: ``experiments/Frequency kink in full GPT-2 training.md``. Every job runs through
``results/gptkink/job.sbatch`` (one H100, 23h50 limit, checkpoint on USR1). Run from the
repository root with the system python3 (standard library only; it works on the login
or jump host, where the lm/ venv's python3.12 does not exist; the jobs themselves use
lm/bin/python on the GPU node). Add --dry-run to print the sbatch commands.

The QOS allows one running H100 job per user (five submitted), so runs are not separate
jobs: ``tune`` and ``stage1`` append tasks to results/gptkink/queue/tasks.txt and a
single ``gk_worker`` job (gptkink_worker.py) runs them in order, checkpointing and
resubmitting itself before the 24 h limit.

    python3 experiments/gptkink_launch.py prep      # tokenize OWT (own job, once)
    python3 experiments/gptkink_launch.py smoke     # Stage 0 checks (own job, ~30 min)
    python3 experiments/gptkink_launch.py tune      # queue the lr grid (20 tasks, 10% budget)
    python3 experiments/gptkink_launch.py worker    # submit the worker if none is queued
    python3 experiments/gptkink_launch.py status    # queue, tuning table, runs, probes
    python3 experiments/gptkink_launch.py stage1    # queue 8 runs + probes (after tuning)
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
Q = RES / "queue"

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


def enqueue(tid, cmd, dry):
    """Append a task (one command, no shell operators) unless the id is already listed."""
    f = Q / "tasks.txt"
    f.parent.mkdir(parents=True, exist_ok=True)
    have = {line.split("\t", 1)[0] for line in f.read_text().splitlines()} if f.exists() else set()
    line = f"{tid}\t{' '.join(cmd)}"
    if tid in have:
        print(f"  (queued already) {tid}")
        return
    print(f"  + {line}")
    if not dry:
        with open(f, "a") as fh:
            fh.write(line + "\n")


def cmd_tune(a):
    for b in BS:
        for g in GRID:
            lr = LR0 * (b / B0) ** 0.5 * g
            run = f"tune_b{b // 1024}k_lr{lr:.3g}"
            enqueue(run, train_cmd(run, b, lr, 0.99, TUNE_TOKENS, tune=True), a.dry_run)


def cmd_worker(a):
    jobs = running_jobs()
    if "gk_worker" in jobs:
        print("a gk_worker job is already queued or running")
        return
    others = sorted(j for j in jobs if j.startswith("gk_"))
    if others:
        print(f"note: {others} queued/running; the worker waits for them (1 running job per user)")
    sbatch("gk_worker", [PY, "experiments/gptkink_worker.py"], a.dry_run)


def queue_status():
    f = Q / "tasks.txt"
    if not f.exists():
        return
    ids = [line.split("\t", 1)[0] for line in f.read_text().splitlines() if line.strip()]
    run = (Q / "running").read_text().split("\t")[0] if (Q / "running").exists() else None
    print("== queue (" + ", ".join(f"{k}: {v}" for k, v in (
        ("tasks", len(ids)), ("done", sum((Q / "done" / t).exists() for t in ids)),
        ("failed", sum((Q / "failed" / t).exists() for t in ids)))) + f"; running: {run or '-'})")
    for t in ids:
        if (Q / "failed" / t).exists():
            print(f"  FAILED {t}: {(Q / 'failed' / t).read_text().strip()}  (log: results/gptkink/queue/logs/{t}.log)")
    print("== slurm:", " ".join(sorted(j for j in running_jobs() if j.startswith("gk_"))) or "-")


def cmd_status(a):
    queue_status()
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


def cmd_stage1(a):
    best = tuned_lrs()
    missing = [b for b in BS if b not in best]
    if missing and not a.lr_from_rule:
        raise SystemExit(f"no tuning result for b = {missing}; run `tune` + `worker` first (or --lr-from-rule)")
    for b in BS:
        lr = LR0 * (b / B0) ** 0.5 if a.lr_from_rule else best[b]["lr"]
        for beta2 in HEAD_BETA2:
            run = name(b, beta2)
            enqueue(run, train_cmd(run, b, lr, beta2, RUN_TOKENS), a.dry_run)
            enqueue(f"probe_{run}", [PY, "experiments/gptkink_probe.py", "--run", run], a.dry_run)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["prep", "smoke", "tune", "status", "stage1", "worker"])
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--lr-from-rule", action="store_true", help="stage1 without tuning: sqrt rule")
    a = ap.parse_args()
    (RES / "slurm").mkdir(parents=True, exist_ok=True)
    {"prep": cmd_prep, "smoke": cmd_smoke, "tune": cmd_tune, "status": cmd_status,
     "stage1": cmd_stage1, "worker": cmd_worker}[a.cmd](a)


if __name__ == "__main__":
    main()
