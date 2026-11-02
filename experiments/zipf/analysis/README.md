# Analysis scripts for the Zipf experiments

These scripts are numpy-only. They read the `.npz` outputs of `experiments/zipf_scaling.py` and write to
`figs/experiments/zipf/analysis/<experiment>/`. Run them from their own directory, for example
`cd e1_rates && ../../../../real/bin/python fig.py`. They were written during the analysis of each
experiment, and the numbers quoted in `experiments/zipf/README.md` come from them.

| folder | run order | outputs |
|---|---|---|
| `e1_rates` (E1a, `rates.npz`) | `decompose.py init`, `decompose.py mid`, `analysis.py`, `analysis2.py`, `checks.py`, `fig.py` | `decomp_{init,mid}.npz`, `results.json`, `results2.json`, `e1a_rates_summary.png` |
| `e1_timescales` (E1b, `timescales.npz`, `rates.npz`) | `e1b_analysis.py`, `e1b_figure.py`, `frozen_rate_checks.py` | `e1b_numbers.json`, `e1b_summary.png` |

`decompose.py` splits each frozen rate into the forcing `r^2 pi_c E[psi_c(1 - p_c)]`, computed from the stored
psi tables, and the restoring part `-(phi^T O)_cc`. The Gaussian draws are seeded, so reruns reproduce the
saved files.
