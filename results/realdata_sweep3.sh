#!/bin/bash
# Sweep 3: HIGGS (4M rows) and the CIFAR finite-D check (D = 512, 1024; 2048 is in sweep 1).
cd "$(dirname "$0")/.."
export XLA_PYTHON_CLIENT_PREALLOCATE=false
real/bin/python experiments/realdata.py diag higgs_relu+sketch_1024 2>&1 | grep -v Warn
for name in higgs_relu+sketch_1024 cifar_animal_sketch_512 cifar_animal_sketch_1024; do
  for b in "0.9 0.999" "0.9 0.95" "0.0 0.9"; do
    set -- $b
    echo "=== $name beta1=$1 beta2=$2"
    real/bin/python experiments/realdata.py run $name --eta-rel 0.1 --beta1 $1 --beta2 $2 --T 100 \
      --modes real twin_emp twin_gauss 2>&1 | grep -v Warn | grep -E "rel. err|eta_rel|Error|Trace"
  done
done
