#!/bin/bash
# Stage-1 b=1k run with the clip threshold chosen from the no-clip check
# (results/gptkink/check/tune/check_b1k_noclip.json, 1e8 tokens, lr 2.5e-4, no clipping):
#   * check finished and its loss is within 0.05 of the clipped tuning run (4.513): no clipping;
#   * otherwise clip at 2x the check's median gradient norm (spikes only), or at 1.0 if unknown.
# Usage: bash experiments/gptkink_b1k.sh <run name> <head beta2>
# Ends with exec so the worker's USR1 reaches the python process.
cd "$(dirname "$0")/.."
CLIP=$(lm/bin/python - <<'PY'
import json, math
from pathlib import Path
f = Path("results/gptkink/check/tune/check_b1k_noclip.json")
ref = json.loads(Path("results/gptkink/tune/tune_b1k_lr0.00025.json").read_text())["final_loss"]
if not f.exists():
    print(1.0); raise SystemExit
d = json.loads(f.read_text())
loss, g = d.get("final_loss"), (d.get("gnorm_med_p90_max") or [float("nan")])[0]
if loss is not None and math.isfinite(loss) and loss <= ref + 0.05:
    print(1e9)
elif g is not None and math.isfinite(g):
    print(f"{2 * g:.3g}")
else:
    print(1.0)
PY
)
echo "b=1k clip threshold: $CLIP (from the no-clip check)"
exec lm/bin/python experiments/gptkink_train.py --name "$1" --B 4 --lr 0.00025 --head-beta2 "$2" --tokens 1e+09 --clip "$CLIP"
