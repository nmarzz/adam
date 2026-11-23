#!/bin/bash
# Frozen-drift follow-ups: centred features (beta2 = 0.99) and the beta2 sweep (raw features).
cd "$(dirname "$0")/../.."
export XLA_PYTHON_CLIENT_PREALLOCATE=false
for s in frozen_beta2 frozen_center; do
  echo "##### $s $(date)"
  real/bin/python experiments/freqkink.py sweep --spec results/freqkink/$s.json 2>&1 | grep --line-buffered -v -E "Warn|bfc_allocator"
done
echo "SWEEP2 DONE $(date)"
