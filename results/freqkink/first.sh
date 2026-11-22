#!/bin/bash
# First deliverable of the frequency-kink plan: beta2 = 0.99, b in {32,128,512,2048},
# eps in {1e-12, 1e-8}, Adam (beta1 = 0, bias correction on) and SGD, raw GPT-2 embeddings.
# lr tuned per configuration on a 4M-sample budget, then one full 40M-sample pass.
cd "$(dirname "$0")/../.."
export XLA_PYTHON_CLIENT_PREALLOCATE=false
real/bin/python experiments/freqkink.py sweep --spec results/freqkink/first.json 2>&1 | grep --line-buffered -v Warn
real/bin/python experiments/freqkink_analysis.py --rate G --tw 100
real/bin/python experiments/freqkink_analysis.py --rate Gt --theta 1
real/bin/python experiments/freqkink_analysis.py --rate half
