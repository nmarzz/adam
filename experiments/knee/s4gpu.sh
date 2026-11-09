#!/bin/bash
# Series J stage 4, GPU part: frozen field on S3a states (eps = 1e-8), then the S3b gate group.
set -u
cd "$(dirname "$0")/../.."
PY=real/bin/python
date
(cd experiments/knee/analysis && ../../../$PY frozen_field.py)
date
echo "FIELD DONE"
# S3b gate (plan: kappa = 1, alpha = 1 reproduces series I's C = 128 fixed point and t_c)
$PY experiments/knee.py run --classes 128 --alphas 1 --kappas 1 --eps 1e-8 --opts adam --unroll 4 --runs 32 --dtype f64
$PY experiments/knee.py run --classes 128 --alphas 1 --kappas 1 --opts sgd --unroll 1 --runs 32 --dtype f64
date
echo "S4GPU DONE"
