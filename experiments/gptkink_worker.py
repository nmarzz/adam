"""Sequential task worker for the gptkink experiment (one H100 job per user).

The QOS allows one running job per user (and five submitted), so all work goes through a
task list, ``results/gptkink/queue/tasks.txt`` (one ``<id>\\t<shell command>`` per line,
appended by ``gptkink_launch.py``), and a single worker job runs the tasks in order:

* a task is done when its command exits 0 without the worker having been signalled
  (marker ``queue/done/<id>``); a failed task is marked ``queue/failed/<id>`` and skipped;
* slurm sends USR1 five minutes before the time limit (job.sbatch); the worker forwards
  it to the running task, which checkpoints and exits, then submits a new worker job and
  exits. The new job resumes that task from its checkpoint;
* when no task is left, the worker exits.

Run only through ``python3 experiments/gptkink_launch.py worker``.
"""
from __future__ import annotations

import os
import signal
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
Q = Path(os.environ.get("GPTKINK_QUEUE", ROOT / "results" / "gptkink" / "queue"))


def tasks():
    out = []
    f = Q / "tasks.txt"
    if f.exists():
        for line in f.read_text().splitlines():
            if line.strip() and not line.startswith("#"):
                tid, cmd = line.split("\t", 1)
                out.append((tid, cmd))
    return out


def pending():
    return [(t, c) for t, c in tasks() if not (Q / "done" / t).exists() and not (Q / "failed" / t).exists()]


def resubmit():
    r = subprocess.run(["sbatch", "--parsable", "--job-name", "gk_worker", "results/gptkink/job.sbatch",
                        "lm/bin/python", "experiments/gptkink_worker.py"],
                       cwd=ROOT, capture_output=True, text=True)
    print(f"resubmitted worker: {r.stdout.strip() or r.stderr.strip()}", flush=True)


def main():
    for d in ("done", "failed", "logs"):
        (Q / d).mkdir(parents=True, exist_ok=True)
    state = {"stop": False, "child": None}

    def on_signal(*_):
        state["stop"] = True
        if state["child"] is not None and state["child"].poll() is None:
            state["child"].send_signal(signal.SIGUSR1)

    signal.signal(signal.SIGUSR1, on_signal)
    signal.signal(signal.SIGTERM, on_signal)
    job = os.environ.get("SLURM_JOB_ID", "local")
    while not state["stop"]:
        todo = pending()
        if not todo:
            print("queue empty", flush=True)
            return
        tid, cmd = todo[0]
        print(f"[{time.strftime('%c')}] job {job}: task {tid}: {cmd}", flush=True)
        (Q / "running").write_text(f"{tid}\t{job}\t{time.strftime('%c')}\n")
        with open(Q / "logs" / f"{tid}.log", "a") as log:
            log.write(f"\n### job {job} {time.strftime('%c')}\n")
            log.flush()
            state["child"] = subprocess.Popen(["bash", "-c", "exec " + cmd], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                              env=dict(os.environ, PYTHONUNBUFFERED="1"))
            rc = state["child"].wait()
            state["child"] = None
        (Q / "running").unlink(missing_ok=True)
        if state["stop"]:
            print(f"signalled during {tid} (exit {rc}); it resumes in the next worker", flush=True)
            break
        if rc == 0:
            (Q / "done" / tid).write_text(f"{job} {time.strftime('%c')}\n")
            print(f"  done {tid}", flush=True)
        else:
            (Q / "failed" / tid).write_text(f"exit {rc}, job {job} {time.strftime('%c')}\n")
            print(f"  FAILED {tid} (exit {rc}); see queue/logs/{tid}.log", flush=True)
    if pending() and not os.environ.get("GPTKINK_NO_RESUBMIT"):
        resubmit()


if __name__ == "__main__":
    main()
