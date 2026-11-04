# Series I: class imbalance, Adam vs SGD — per-class and risk convergence rates — log

Plan: `/home/math/noah.marshall2@MAIL.MCGILL.CA/.claude/plans/i-want-to-explore-memoized-iverson.md`. Branch `zipf-scaling`.
Hardware: one H100 NVL, JAX (`real/` venv), float64. Run everything from the repository root with
`real/bin/python experiments/imbalance.py <command>`. Outputs go to `figs/experiments/imbalance/` (untracked):
`ode/`, `sim/`, `frozen/`, `analysis/` (figures and `numbers.json`), `logs/`, and `runs.jsonl` (one line per run,
with wall time, steps and git hash).

## Goal and models

Compare Adam with SGD under Zipf class imbalance:
- how each class and the total risk converge;
- whether the per-class rate shows the kink of E1 (`experiments/zipf/README.md`). E1 found frozen Adam drift
  `∝ sqrt(b pi)` above `pi ≈ kappa_beta (1-beta2)/(b F^2)` and `∝ pi` below; SGD's drift is `∝ pi` everywhere.

| | |
|---|---|
| classes | `C ∈ {8, 32}`, `pi_c ∝ c^{-alpha}`, `alpha ∈ {1, 2}` |
| data | `x = r q_c + sqrt(kappa) xi`, `r = 5`, `q_c` orthonormal, `kappa ∈ {1, 0.1}` (`K = kappa Id`) |
| model m0 | softmax CE, `C` logits, no bias. **ODE** (collapsed isotropic; psi route for Adam, path-free moments for SGD) and direct simulation |
| model m1 | m0 plus a learned bias `beta ∈ R^C`, same optimizer and `lr`, `beta_0 = 0`. **Simulation only** |
| optimizers | Adam `beta1 = 0.9`, `beta2 ∈ {0.9, 0.95, 0.99}`, `eps = 1e-8` (K-scaled), no bias correction; SGD. `b = 1` |
| learning rates | Adam `eta = 0.3`, SGD `eta = 3` (ODE units, from E-series `tune`; the same `eta` for the bias) |
| start | ODE: `B_0 = 0.01 kappa Id`, `O_0 = 0` (the `D -> infinity` limit of `Theta_0 = 0.1 N(0, 1/D)`). Simulation: `Theta_0` at `D` |
| simulations | `D = 1000`, 16 runs, record grid `0.01 · 10^{k/16}` |
| ODE | step factor `safety = 1` (as E2–E4), the same nested grid, horizons `T = 10^3` (SGD at `C = 32`: `10^3.5`) |

Dropped for now: `C = 128`, `beta2 = 0.999`, and an Adam-bias ODE.
- **Why no Adam-bias ODE.** A bias coordinate is normalised by EMAs of `f` with no `xi` factor, so its update is
  sign-like. For a rare class, each own event dominates `v_k` for about 500–1400 steps at `beta2 = 0.99` and
  switches off the background push. The psi route would sample this from only about `pi_k 2^18` events, so the
  ODE needs a new estimator.
- **The SGD-bias ODE would be trivial**: `dbeta/dt = -eta (E p - pi)`.

### The low-noise branch is an exact rescaling

Rescale `x' = x / sqrt(kappa)` and `Theta' = sqrt(kappa) Theta`. Then the logits, `B = Theta^T K Theta` and
`O = Theta^T Xbar` are unchanged, the means have norm `r' = r / sqrt(kappa)` (15.8 at `kappa = 0.1`), and:
- SGD with `eta` becomes SGD with `eta' = kappa eta`;
- Adam with `eta` becomes Adam with `eta' = sqrt(kappa) eta`, and its K-scaled `eps` is invariant.

So the ODE for `kappa = 0.1` is the `K = Id` ODE with `r'` and `eta'`.
- **The bias does not rescale.** Its gradient `f` has no `x` factor, so in primed units the bias keeps `eta` while
  the weights get `eta'`. Relative to the weights, the bias learns `1/kappa` (SGD) or `1/sqrt(kappa)` (Adam) times
  faster at `kappa = 0.1`.
- **Finite D.** The ODE drops the per-coordinate mean energy `r'^2/D` against `1`, which is 0.25 at `D = 1000` for
  `kappa = 0.1`. So the `kappa = 0.1` simulations at `D = 1000` are expected to deviate more, and `D = 4000` checks
  are included.

### Theory reference values (`b = 1`, from `analysis/lib.py`)

| `beta2` | 0.9 | 0.95 | 0.99 |
|---|---|---|---|
| `kappa_beta` | 0.414 | 0.922 | 1.432 |
| `kappa_beta (1 - beta2)` | 0.041 | 0.046 | 0.014 |
| `F` (C = 8 / 32 / 128) | 0.653 / 0.893 / 0.971 | 0.740 / 0.958 / 0.995 | 0.585 / 0.919 / 0.991 |
| kink with `F`, C = 8 / 32 | 0.097 / 0.052 | 0.084 / 0.050 | 0.042 / 0.017 |
| `coef_beta = c_beta sqrt(2/pi)/(1 - beta1/sqrt(beta2))` | 4.92 | 4.66 | 8.36 |

`F` sums all lags until `rho^l < 1e-8`. At `C = 128` it reproduces E1a's values (0.995, 0.991).

| `alpha` | C = 8 pi range | C = 32 pi range |
|---|---|---|
| 1 | 0.37 – 0.046 | 0.25 – 7.7e-3 |
| 2 | 0.66 – 1.0e-2 | 0.62 – 6.1e-4 |

Predictions to test:
- **Adam/SGD learning-time ratio at `Theta_0`, with no free parameters.** It is `10 sqrt(pi)(1 - 1/C) sqrt(kappa)`
  above the kink and `10 (1 - 1/C) sqrt(kappa) / (coef_beta F)` below; the two asymptotes meet at the kink. This
  ratio is the primary trajectory estimator of the kink, because it cancels the common steepening along
  trajectories that E1b found.
- **`beta2 = 0.9` vs `0.95`.** The kinks nearly coincide (`F`-corrected ratio 1.15 at `C = 32`; the naive
  `(1 - beta2)` law gives 2). On the rare side the predicted `t_c(0.9)/t_c(0.95)` is 0.947, and 1.79 for 0.95/0.99.
- **At `C = 8` the background term in `F` dominates.**
- **At the deterministic start the frozen drift does not depend on `alpha`.** Every input sees `p = 1/C`, so the
  `alpha = 1` and `alpha = 2` classes fall on one curve.

## Stage 0: engineering (2026-11-04)

### Code

| Where | What |
|---|---|
| `simulate.py` | `bias=` in `_mixture_grad`, `build_mixture_adam/sgd` and `run_mixture_at(_many)`. Parameters are `(D+1, M)` with the bias as the last row (`x_{D+1} = 1`), and its Adam stability is `eps`. `run_mixture_at` returns `(B, O, beta)`. The default path is unchanged. |
| `problems.py` | `softmax_class_metrics(..., bias=)`: `q \| c ~ N(O[:, c] + bias, B)` |
| `experiments/zipf_scaling.py` | `zipf_model(..., noise=kappa)`; `bias=` in `simulate_runs` and `class_metrics` |
| `experiments/imbalance.py` | Driver with subcommands `selftest ode sim frozen frozen-bias probe analyze plot recheck`. It names and caches runs, extends ODE runs, and computes frozen drifts (psi route, 4 plan seeds). Bias drifts come from direct `q`-space chains of `E[m/sqrt(v + eps)]`. |
| `experiments/imbalance/selftest.py` | Self-tests (below) |
| `experiments/imbalance/analysis/` | `lib.py` (theory, measures, fits; reuses `e1b_lib`), `analyze.py` (tables, `numbers.json`), `figures.py` (Figs A–E), `recheck.py` (independent recomputation) |

### Self-tests (`real/bin/python experiments/imbalance.py selftest`, 227 s): 8/8 pass

The full run gave 6/7. The rescaling test was then corrected (see the end of this section) and its two checks
were re-run alone. `tests/test_zipf_coefficients.py` still passes after the simulator change: 15/15 in 339 s.

| Check | Result |
|---|---|
| `bias=False` simulation vs `simulate.py` at `HEAD`, Adam and SGD, `b ∈ {1, 4}` | bit-identical (max diff 0) |
| Bias gradient at `beta = 0`: weight rows vs the no-bias gradient; bias row vs `mean f` | 0, 0 |
| SGD conserves `sum(beta)` | 6e-16 (with `max |beta| = 0.13`) |
| Adam stability on the bias row | `eps` |
| `K = kappa Id` vs the primed `K = Id` problem (no bias) | Adam 2.5e-15, SGD 1.1e-15 relative |
| The same with a bias, giving the bias row the unscaled `eta` | Adam 5.4e-15, SGD 1.7e-15 |
| ODE extension (run to 10, then extend to 100) vs a direct run to 100, Adam and SGD, `kappa ∈ {1, 0.1}` | identical (0) |
| Regression: series-I ODE vs the cached E2 runs `{adam_..._b20.99, sgd}_C64_a2` (`t <= 100`) | `|dB|` 9e-14 / 2e-14; `|d ell|` 4e-15 / 3e-15 |

The first version of the rescaling test applied `eta'` to the bias row too and failed (0.7–0.9 relative). This is
the bias non-invariance described above, not a code error.

### Timings
- **Simulation:** about 90–110 µs per step for 16 vmapped runs at `D = 1000` (C = 8 and 32, m0 and m1, Adam and
  SGD; includes compile). That is about 100 s per `10^6`-step config.
- **ODE, C = 8, `T = 100`:** Adam 152 steps (`kappa = 1`) and 113 steps (`kappa = 0.1`); SGD 119 and 65 steps; a
  few seconds each after compile.
