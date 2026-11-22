#!/bin/bash
# Frozen-drift runs (lr = 1e-10): stationary Adam denominators with W ~ 0, so the
# own-event gain G grows at the per-class drift at initialization.
cd "$(dirname "$0")/../.."
export XLA_PYTHON_CLIENT_PREALLOCATE=false
real/bin/python experiments/freqkink.py sweep --spec results/freqkink/frozen.json 2>&1 | grep --line-buffered -v -E "Warn|bfc_allocator"
real/bin/python experiments/freqkink_analysis.py --rate Gslope --frozen
