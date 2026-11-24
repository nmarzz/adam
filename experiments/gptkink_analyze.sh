#!/bin/bash
# Stage-1 analysis (last task of the queue): own-event forcing (gain_own_sum) and net drift
# (gain_sum) of the probes at every checkpoint fraction, pooled over b and head beta2;
# full-run curves (time for G to reach 1 nat); tuning figure. Output: figs/gptkink/, and
# results/gptkink/probes/frac_*/analysis_*.json.
cd "$(dirname "$0")/.."
PY=real/bin/python
for d in results/gptkink/probes/frac_*; do
  f=$(basename "$d")
  for key in gain_own_sum gain_sum; do
    FREQKINK_GAIN=$key FREQKINK_OUT=$d FREQKINK_FIGS=figs/gptkink/$f \
      $PY experiments/freqkink_analysis.py --rate Gslope --frozen --norm plateau --pool --tag _$key > /dev/null \
      && echo "analysed $f ($key)"
  done
done
FREQKINK_OUT=results/gptkink FREQKINK_FIGS=figs/gptkink/runs $PY experiments/freqkink_analysis.py --rate Gt --theta 1 > /dev/null && echo "analysed full runs"
$PY experiments/gptkink_tune_plot.py
