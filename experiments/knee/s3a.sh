#!/bin/bash
# Series J stage 3a: alpha = 0, C = 128 and 32, all kappa, eps in {1e-8, 1e-16}, + checks.
set -u
cd "$(dirname "$0")/../.."
PY=real/bin/python
K="2 1 0.5 0.25 0.1"
common="--alphas 0 --kappas $K --runs 32 --dtype f64 --extend-until-certified"
for C in 128 32; do
  $PY experiments/knee.py run --classes $C --opts adam --unroll 4 --eps 1e-8 1e-16 $common
  $PY experiments/knee.py run --classes $C --opts sgd  --unroll 1 $common
done
# finite-D check: D halved at kappa = 0.1 (5000 -> 2500), C = 128, eps = 1e-8
$PY experiments/knee.py run --classes 128 --alphas 0 --kappas 0.1 --dimension 2500 --runs 32 \
    --dtype f64 --opts adam --unroll 4 --eps 1e-8 --extend-until-certified
# high-dimensional check of the beta2 = 0.999 lane at C = 32, kappa = 1 (D = 4000; judge on t >= 3)
$PY experiments/knee.py run --classes 32 --alphas 0 --kappas 1 --dimension 4000 --runs 32 \
    --dtype f64 --opts adam --unroll 4 --eps 1e-8 --beta2s 0.999 --extend-until-certified
echo "S3A DONE"
