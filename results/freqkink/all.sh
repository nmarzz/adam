#!/bin/bash
# Tuned sweep, then frozen-drift sweep, then analyses. Finished runs and cached tuning are skipped.
cd "$(dirname "$0")/../.."
results/freqkink/first.sh >> results/freqkink/first.log 2>&1
results/freqkink/frozen.sh >> results/freqkink/frozen.log 2>&1
echo "ALL DONE $(date)" >> results/freqkink/first.log
