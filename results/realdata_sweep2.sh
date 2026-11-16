#!/bin/bash
# Sweep 2: preprocessed (standardize+sphere) candidate feature sets, eta_rel = 0.1, three (beta1, beta2) settings.
cd "$(dirname "$0")/.."
export XLA_PYTHON_CLIENT_PREALLOCATE=false
for name in owt_logfreq_sketchzs_768 msd_year_relu+sketchs_1024; do
  for b in "0.9 0.999" "0.9 0.95" "0.0 0.9"; do
    set -- $b
    echo "=== $name beta1=$1 beta2=$2"
    real/bin/python experiments/realdata.py run $name --eta-rel 0.1 --beta1 $1 --beta2 $2 --T 100 \
      --modes real twin_emp twin_gauss 2>&1 | grep -v Warn | grep -E "rel. err|eta_rel|Error|Trace"
  done
done
