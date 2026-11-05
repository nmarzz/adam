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

## Stage 1: C = 8 (2026-11-04)

```
real/bin/python experiments/imbalance.py ode --classes 8            # 16 ODE runs, 20-30 s each
real/bin/python experiments/imbalance.py frozen --classes 8         # 30 s
real/bin/python experiments/imbalance.py frozen-bias --classes 8    # 3 min
real/bin/python experiments/imbalance.py sim --classes 8            # 32 configs x 16 runs x 1e6 steps, ~31 s each
real/bin/python experiments/imbalance.py sim --classes 8 --alphas 2 --models m0 --opts adam --beta2s 0.99 \
    --dimension {2000,4000} --sim-horizon 100                       # finite-D check
real/bin/python experiments/imbalance.py analyze; real/bin/python experiments/imbalance.py plot
```

Simulation at `D = 1000` takes about 31 µs per step for 16 runs, three times faster than the Stage 0 probe, which
included compilation. Tables: `figs/experiments/imbalance/analysis/tables.md`. Figures: `A_*`–`E_*` in the same
folder.

### Risk (m0, ODE; `R(0) = 2.08`)

| `kappa` | `alpha` | opt | `t(R/2)` | `t(R/10)` | `-dlogR/dlogt` at t = 1 / 10 / 100 / 1000 | `R(1000)` |
|---|---|---|---|---|---|---|
| 1 | 1 | Adam 0.9 / 0.95 / 0.99 | 0.44 / 0.44 / 0.43 | 2.0 / 2.1 / 1.9 | 1.07 / 1.23 / 0.08 / −0.11 (0.9) | 5.5e-3 / 5.3e-3 / 6.3e-3 |
| 1 | 1 | SGD | 0.14 | 1.2 | 0.91 / 0.72 / 0.42 / 0.11 | 5.2e-3 |
| 1 | 2 | Adam 0.9 / 0.95 / 0.99 | 0.22 / 0.23 / 0.23 | 1.6 / 1.6 / 1.5 | 0.87 / 1.36 / 0.20 / −0.08 (0.9) | 3.6e-3 / 3.5e-3 / 4.1e-3 |
| 1 | 2 | SGD | 0.058 | 0.97 | 0.70 / 0.80 / 0.49 / 0.16 | 3.5e-3 |
| 0.1 | 1 | Adam (all `beta2`) | 0.12–0.13 | 0.42–0.45 | ~5 / 1.2 / 1.0 / 0.98 | 4e-8 |
| 0.1 | 1 | SGD | 0.12 | 0.78 | 1.20 / 1.05 / 0.99 / 0.98 | 1e-4 |
| 0.1 | 2 | Adam (all `beta2`) | 0.066–0.069 | 0.34–0.37 | ~2 / 1.6–2.0 / 1.0 / 0.98 | 4e-8 |
| 0.1 | 2 | SGD | 0.053 | 0.67 | 0.87 / 1.13 / 1.01 / 0.99 | 1e-4 |

All 8 classes are learned in every run.
- **`K = Id`:**
  - SGD reaches `R/2` first: `eta_SGD = 3` against 0.3 for Adam.
  - Adam overtakes in the middle, with a local exponent of 1.2–1.4 against SGD's 0.7–0.8.
  - By `t ~ 100` Adam has flattened onto a floor. Its risk then **rises slightly** (exponent −0.07 to −0.11 at
    `t = 1000`, `beta2 = 0.99` worst), and its last-decade class slopes are negative for the frequent classes.
  - SGD is still descending at `t = 1000`.
- **`K = 0.1 Id`** (separable): both reach an exponent of about 1 (`ell ~ 1/t`, as for logistic regression on
  separable data), but Adam's prefactor is about 2500 times smaller (`R(1000)` 4e-8 against 1e-4).

### Kink in the frozen drift at `Theta_0`

Fits pool `alpha = 1` and `alpha = 2`, 16 classes per `beta2`.

| `beta2` | free SBPL slopes (rare / frequent) | free `pi_b / kink_F` | fixed (1, ½) `pi_b / kink_F` | `alpha` overlap |
|---|---|---|---|---|
| 0.9 | 0.99 / 0.49 | 1.15 | 1.06 | 0.4% |
| 0.95 | 0.99 / 0.50 | 1.08 | 1.05 | 0.6% |
| 0.99 | 0.99 / 0.51 | 1.07 | 1.05 | 0.2% |

- **The law of E1 holds at `C = 8`, but only with `F`.** `F = 0.65`, 0.74 and 0.59, so without it the predicted
  kinks would be 2.3, 1.8 and 2.9 times too small.
- **The background term `b/C^2` dominates `F`.** This is the first measurement of `F` at small `C`; E1 measured it
  at `C = 128`.
- **SGD** equals `r^2 pi (1 - 1/C)` to 4 digits.
- **`kappa = 0.1`** gives identical numbers, as the rescaling predicts.

### Kink along the trajectories (`t_c` at ½)

| quantity | `K = Id` | `K = 0.1 Id` |
|---|---|---|
| Adam/SGD ratio break `pi_b / kink_F` (m0) | 1.11, 1.23, 1.65 (`alpha` = 1); 1.03, 1.13, 0.87 (`alpha` = 2) | 1.11, 1.24, 1.88; 0.60, 0.68, 0.78 |
| ratio level / predicted rare plateau `10 (1-1/C) sqrt(kappa)/(coef_beta F)` | 0.73–1.14 | 0.63–1.25 |
| free `t_c` broken line, slopes above / below the break (m0) | −0.71 to −0.78 / −0.82 to −1.16 | −0.65 to −0.71 / −0.67 to −0.79 |

The `beta2`-pair ratio `t_c(0.9)/t_c(0.95)` on the rare side (π < kink_F/3; m0, `alpha = 2`, 4 classes) is
1.003 (`K = Id`) and 0.982 (`K = 0.1 Id`). The predictions are 0.947 with the full `coef_beta`, and 1.41 for the
plan's `1/sqrt(1-beta2)` rare law. The window ratio is 0.97.

- **The Adam/SGD ratio** places the break within 0.6–1.9 of `kink_F` and its level within 0.6–1.25 of the
  zero-parameter prediction.
- **Eight classes give only 2–5 classes per side.** These are order-of-magnitude confirmations; `C = 32` is the
  real test.
- **`beta2 = 0.9` and 0.95 are indistinguishable**, as `kappa_beta` predicts (equal kinks, rare prefactors within
  5%). The naive `(1 - beta2)` scaling is ruled out.
- **At `K = 0.1 Id` the `t_c` slopes are shallower on both sides.** Learning is fast relative to the Adam memory, so
  half-loss times fall in `t < 1`.

### ODE vs simulation (m0): time lag `t_sim(R)/t_ode(R)` at fixed risk, `t >= 1`

| | D = 1000 | D = 2000 | D = 4000 |
|---|---|---|---|
| `K = Id`, Adam (all `beta2`, `alpha`) | 1.018–1.038 (median) | 1.012 (`alpha = 2`, 0.99) | 1.002 |
| `K = Id`, SGD | 0.995–1.006 | | |
| `K = 0.1 Id`, Adam | 1.11–1.35 | 1.16 | 1.06 |
| `K = 0.1 Id`, SGD | 0.999 | | |

The sup relative risk gap first used here is misleading for `K = 0.1 Id`. It reaches 170–1460% only because the
risk falls 4 decades between `t = 1` and 10, so a 30% time lag becomes a factor of 4.6 in `R`. The time lag at
fixed risk is the metric used from now on.

- **The gap is finite-D and closes as `1/D`.** The excess lag is 2.5%, 1.5% and 0.4% (`K = Id`) and 29%, 16% and
  6% (`K = 0.1 Id`) at `D` = 1000, 2000 and 4000.
- **Its 10 times larger coefficient at `K = 0.1 Id`** matches the mean energy per coordinate `r'^2/D` that Adam's
  normalisation sees and the ODE drops (`r'^2 = 250` against 25).
- **SGD has no such term.** It agrees at 0.1%.
- **The ODE is correct in the limit.** Absolute times of `D = 1000` simulations at `K = 0.1 Id`, including all of
  M1 there, are about 30% slow. Comparisons at equal `D` (m1 against m0) are unaffected.
- **Before `t = 1`** the simulations run ahead: Adam's warm-up without bias correction, as in E0. This also closes
  with `D`.

### Bias model (m1, simulation)

| | `K = Id` | `K = 0.1 Id` |
|---|---|---|
| median `t_c(m1)/t_c(m0)` | 0.96–1.07 | 0.98–1.05 |
| `R(1000)`, `t(R/2)` | equal to m0 within 1–4% | equal |
| learned bias at `T`: corr / slope of `beta - mean` against `log pi - mean` | Adam +0.84 to +1.00 / 0.11–0.31 (largest at `beta2 = 0.99`); SGD +0.91 to +0.96 / 0.06 | Adam −0.8 to −0.99 / −0.01; SGD +1.00 / 0.03 |

**Frozen bias drift at the initial state** (chains, at least 1.4e6 own events per class; positive values push
`beta_c` down):
- **Adam is sign-like.**
  - Class 1: −0.43 to −0.74 (SGD −0.24 to −0.53).
  - Rarest class: +0.37 to +0.86 (SGD +0.08 to +0.11).
  - The sign changes at `pi ≈ 0.11–0.15`, at the same place as SGD (`1/C = 0.125`).
- **At the calibrated state `beta = log pi`:**
  - SGD's drift vanishes, as it must (`E p = pi`).
  - Adam keeps pushing the rare biases down: +0.54, +0.35 and +0.06 at the rarest class for `beta2` = 0.9, 0.95
    and 0.99 (`alpha = 2`).
  - This is the predicted over-suppression of rare biases. It is strongest at short memory, where own events stop
    dominating `v`.

At `C = 8` the bias barely changes per-class learning (within ±7%). The weights carry the class structure, and the
learned biases move only 6–31% of the way towards the Bayes offsets `log pi` by `t = 1000`. At `K = 0.1 Id` the
data are separable and the bias does nothing measurable.

**Checkpoint 1.**
- The frozen kink law, including `F`, holds at `C = 8`, and `kappa_beta` is confirmed by the 0.9/0.95 pair.
- The trajectory kink is resolved only roughly with 8 classes.
- The ODE is validated at `D = 1000` for `K = Id` (≤ 4% time lag, closing as `1/D`). At `K = 0.1 Id` the
  simulations carry a 30% finite-D lag.
- No stall at `C = 8`, but Adam's risk rises late at `K = Id`.

## Stage 2: C = 32 (2026-11-04)

```
real/bin/python experiments/imbalance.py ode --classes 32             # 16 runs, 11-72 s each
real/bin/python experiments/imbalance.py frozen --classes 32          # 50 s
real/bin/python experiments/imbalance.py frozen-bias --classes 32     # 85 s
real/bin/python experiments/imbalance.py sim --classes 32             # 32 configs (SGD at alpha = 2 to 10^3.5)
real/bin/python experiments/imbalance.py sim --classes 32 --alphas 2 --models m0 --beta2s 0.99 \
    --dimension 4000 --sim-horizon 300                                # finite-D check
real/bin/python experiments/imbalance.py ode --classes 32 --alphas 2 --beta2s 0.9 --opts adam --kappas 1 --safety 0.5
```

Stage 2 took 50 min of wall time. Stages 1 and 2 together, including the regression suite, took about 1.3 h of GPU.

### Risk (m0, ODE; `R(0) = 3.47`)

Ratio `R_Adam / R_SGD` at matched times `t = 0.1 / 1 / 10 / 100 / 1000`:

| `kappa` | `alpha` | `beta2 = 0.9` | 0.95 | 0.99 |
|---|---|---|---|---|
| 1 | 1 | 1.2 / 1.5 / 2.1 / 1.5 / 1.5 | 1.2 / 1.6 / 2.1 / 1.3 / 1.3 | 1.2 / 1.5 / 1.2 / 0.88 / 1.2 |
| 1 | 2 | 1.8 / 1.5 / 1.3 / 2.3 / 2.9 | 1.8 / 1.5 / 1.3 / 2.2 / 3.4 | 1.8 / 1.5 / 1.1 / 1.5 / 1.9 |
| 0.1 | 1 | 1.0 / 0.68 / 2e-3 / 4e-4 / 4e-4 | 1.0 / 0.71 / 2e-3 / 4e-4 / 4e-4 | 1.0 / 0.55 / 7e-4 / 4e-4 / 4e-4 |
| 0.1 | 2 | 1.1 / 0.62 / 0.31 / 1e-3 / 4e-4 | 1.2 / 0.64 / 0.33 / 1e-3 / 4e-4 | 1.2 / 0.55 / 0.19 / 4e-4 / 4e-4 |

The same ratios at `C = 8`, `K = Id`: 1.5–2.1 at `t <= 1`, 0.5–0.8 at `t = 10-100`, 1.0–1.2 at `t = 1000`.

- **Noise decides the winner.**
  - At `K = Id`, `C = 32`, SGD's risk is below Adam's at essentially every time (by up to 3.4x at `alpha = 2`).
  - At `C = 8` Adam leads in the middle.
  - At `K = 0.1 Id` Adam's risk collapses by 4–5 decades in a short window (`t ≈ 1–10` at `alpha = 1`, `10–50`
    at `alpha = 2`), then follows `1/t` about 2700 times below SGD. SGD decays as a power law with exponent about
    1 throughout.
- **Imbalance shifts the comparison towards SGD at `K = Id`.** Going from `alpha = 1` to 2 and from `C = 8` to 32
  raises `R_Adam/R_SGD` late in training. At `K = 0.1 Id` it only delays Adam's collapse (rare classes are learned
  last).
- **The learning rates were tuned once**, at `K = Id`, `C = 32`, `alpha = 1.5` (E-series), where Adam and SGD tied
  at `t = 100`. The comparison is at those fixed rates.

### Learned classes and the C = 32 "stall" (`K = Id`, `alpha = 2`)

`beta2 = 0.9` learns 27/32 classes by `t = 1000`, 0.95 learns 25/32, and 0.99 learns all 32; SGD learns all 32.
- **It is not E4's fixed point.**
  - The unlearned classes are still descending at `t = 1000` (last-decade slopes 0.09–0.30).
  - The relative change of the gaps over `[T/sqrt(10), T]` is 0.10–0.11, against about 3e-4 at E4's fixed points.
  - The rarest class first rises above its initial loss (`ell_32(100)/ell_32(0) = 1.21` at 0.9, 1.16 at 0.95) and
    then recovers.
- **Shorter memory is worse**, the same ordering as E4.
- **At `K = 0.1 Id` every run learns every class.** The slowdown of rare classes needs the data noise. The bias
  model behaves the same way: 27 and 24 classes learned.

### Kink in the frozen drift at `Theta_0` (pooled over `alpha`, 64 classes)

| `beta2` | free slopes (rare / frequent) | free `pi_b / kink_F` | fixed (1, ½) `pi_b / kink_F` | `alpha` overlap |
|---|---|---|---|---|
| 0.9 | 1.00 / 0.50 | 1.05 | 1.04 | 0.5% |
| 0.95 | 1.00 / 0.50 | 1.02 | 1.04 | 0.5% |
| 0.99 | 1.00 / 0.50 | 1.00 | 1.03 | 0.8% |

`kappa = 0.1` gives identical numbers. Across `C = 8` and `C = 32` (6 configs) every frozen kink is within 0–15% of
`kappa_beta (1 - beta2)/(b F^2)`.

### Kink along the trajectories (`t_c` at ½, `C = 32`)

**Adam/SGD ratio break, `pi_b`:**

| `alpha` | `kappa` | `beta2 = 0.9` | 0.95 | 0.99 | `pi_b / kink_F` |
|---|---|---|---|---|---|
| 1 | 1 | 0.092 | 0.076 | 0.027 | 1.77 / 1.52 / 1.58 |
| 2 | 1 | 0.099 | 0.104 | 0.033 | 1.90 / 2.08 / 1.95 |
| 1 | 0.1 | 0.045 | 0.044 | 0.021 | 0.86 / 0.88 / 1.26 |
| 2 | 0.1 | 0.030 | 0.030 | 0.012 | 0.57 / 0.61 / 0.69 |

Other measures:
- **Ratio level against the predicted rare plateau** `10 (1-1/C) sqrt(kappa)/(coef_beta F)`: 0.93–1.26
  (`K = Id`) and 0.62–1.18 (`K = 0.1 Id`).
- **Free broken line in `t_c`:** slopes −0.70 to −0.80 above the break and −1.08 to −1.26 below (`K = Id`);
  −0.64 to −0.73 and −0.78 to −0.92 (`K = 0.1 Id`).
- **`beta2`-pair ratios on the rare side:**
  - `t_c(0.9)/t_c(0.95)` = 0.98–1.01 (`K = Id`, 5 classes) and 0.957–0.958 (`K = 0.1 Id`, 18–26 classes), against
    0.947 predicted.
  - `t_c(0.95)/t_c(0.99)` = 1.72 (`K = Id`, `alpha = 2`, 4 classes) and 1.44 (`K = 0.1 Id`, 22 classes), against
    1.795 predicted.
  - The window ratio is 0.95–0.99.

What this shows:
- **The kink moves with `beta2` as predicted.**
  - The ratio of breaks between `beta2 = 0.95` and 0.99 is 2.8 (`alpha = 1`) and 3.2 (`alpha = 2`), against 2.97.
  - 0.9 and 0.95 coincide (ratio 1.2 and 0.95, against 1.03), unlike the naive `(1 - beta2)` law, which predicts 2.
- **Its absolute position along the trajectory sits about 1.5–2x above the frozen prediction at `K = Id`.** The
  rare plateau of the ratio sits at the predicted height (0.93–1.26). So the window branch is what is lowered:
  along the trajectory, frequent classes gain less from Adam's `sqrt(pi)` window than at `Theta_0`.
- **At `K = 0.1 Id` the break sits at or below the prediction (0.6–1.3).** The ratio keeps falling into the tail
  instead of reaching a plateau: Adam's advantage over SGD grows for rarer classes. The rarest class has
  `t_A/t_S ≈ 0.28` at `beta2 = 0.99`, against 0.40 predicted.
- **Collapse.** `(t_A/t_S)/(10 sqrt(pi)(1-1/C) sqrt(kappa))` against `u = pi/kink_F` puts every model, `C`,
  `alpha`, `kappa` and `beta2` on one band that follows `max(1, u^{-1/2})` shifted right by a factor of about 2
  (`D_kink.png`, panel d).
- **Fitting `t_c` alone is less reliable than the ratio.**
  - Free broken lines give slopes −0.7 to −0.8 above and −1.1 to −1.3 below the break, steeper than the frozen
    ½ and 1. E1b found the same steepening.
  - The smooth fixed-slope forms (p = 1, 2) put the break 5–34 times too high at `K = Id`, because they force slope
    −½ on the head.
  - The Adam/SGD ratio cancels this common steepening, which is why it is the primary estimator.

### ODE vs simulation (m0): time lag at fixed risk (median over levels, `t >= 1`)

| | D = 1000 | D = 4000 (`T = 300`, `alpha = 2`, Adam 0.99 and SGD) |
|---|---|---|
| `K = Id`, Adam | 1.009–1.019 | 0.996 |
| `K = Id`, SGD | 1.002–1.007 | 1.002 |
| `K = 0.1 Id`, Adam | 1.15–1.21 | 1.050 |
| `K = 0.1 Id`, SGD | 0.998 | 1.002 |

Per-class learning times at `K = Id`, D = 1000: mean `log(t_c sim/ODE)` is −0.011 to +0.021 over 22–30 classes per
config. The number of classes learned agrees in every config.

The `C = 8` conclusion holds at `C = 32`. At `K = Id` the ODE is validated to 2% at `D = 1000` and 0.4% at
`D = 4000`. At `K = 0.1 Id` Adam has a finite-D lag of 15–21% at `D = 1000`, and it shrinks 3.6x by `D = 4000`.

**Step size.** Safety 1 against 0.5 (`beta2 = 0.9`, `alpha = 2`, 678 against 1303 steps) differs by at most 0.38%
in any `ell_c(t)`, 0.03% in the risk and 0.02% in any `t_c`, with the same 27 classes learned. The per-class figure
narrowly misses the plan's 0.25%.

### Bias model (m1, simulation, `C = 32`)

**Rare classes (`pi < 3e-3`), geometric mean of `t_c(m1)/t_c(m0)`** (16 runs each, bootstrap 5–95% over runs):

| | Adam 0.9 | Adam 0.95 | Adam 0.99 | SGD |
|---|---|---|---|---|
| `K = Id`, `alpha = 2` | 1.26 [1.23, 1.32] | 1.35 [1.32, 1.44] | 1.10 [1.07, 1.13] | 1.04 [1.02, 1.07] |
| `K = 0.1 Id`, `alpha = 2` | 1.04 [1.00, 1.07] | 1.04 [1.00, 1.07] | 0.99 [0.94, 1.04] | 1.04 [1.01, 1.07] |

**Frozen bias drift**, `alpha = 2`, rarest class (positive pushes `beta_c` down):
- At the initial state: +0.97, +0.96 and +0.91 for Adam (`beta2` = 0.9, 0.95, 0.99), against +0.03 for SGD.
- At the calibrated state `beta = log pi`:
  - Adam: +0.92, +0.86 and +0.58 (median class +0.81, +0.69 and +0.25).
  - SGD: 0, to 4 decimals.
- At `C = 8` the calibrated-state values were +0.54, +0.35 and +0.06.

What this shows:
- **Adding a learned bias slows Adam's rare classes by 10–35% at `K = Id`, most at short memory. SGD is barely
  affected (4%).**
  - The frozen drifts give the mechanism. Adam normalises each bias coordinate by its own EMA of `f^2`, so it keeps
    driving rare biases down even at the Bayes-calibrated offsets, where SGD's drift vanishes.
  - The push weakens as `beta2 -> 1`, where own events dominate `v` for longer.
  - It is far stronger at `C = 32` than at `C = 8` (rarer classes).
- **Final risk and the number of classes learned are unchanged within 1–8%.** At `K = 0.1 Id` the effect
  disappears: the data are separable and the weights do all the work.
- **The learned biases end far from `log pi`.**
  - Adam: slope of `beta - mean` against `log pi - mean` is 0.06–0.32.
  - SGD: −0.03 to −0.02. SGD's bias barely moves, because the weights already calibrate `E p` to `pi` and its
    drift `E p - pi` vanishes.

**Checkpoint 2 / independent recheck.** `recheck.py` reads the raw files and recomputes every `t_c` and `R(T)`
exactly, and every ratio break within 0.7%. Its first version flagged one break (10%). That was its own local
optimiser stopping in a local minimum of the non-smooth hinge objective; the analysis grid value had the lower SSE.
It now seeds the solver from a dense scan.

## Summary (C ∈ {8, 32}, alpha ∈ {1, 2}, beta2 ∈ {0.9, 0.95, 0.99}, b = 1)

| question | answer |
|---|---|
| Risk vs time | At `K = Id` the winner depends on imbalance: Adam leads mid-training at `C = 8`, while SGD is ahead at nearly all times at `C = 32` (up to 3.4x at `alpha = 2`; Adam leads only briefly, at `alpha = 1`, `beta2 = 0.99`, `t ≈ 100`). Adam's risk plateaus and, at `C = 8`, rises slightly late. At `K = 0.1 Id` (separable) Adam collapses the risk by 4–5 decades and ends about 2700x below SGD, both with a `1/t` tail. |
| Per-class rates | Frozen at `Theta_0`: Adam has slope ½ in `pi` above the kink and 1 below; SGD has slope 1 (exact). Along trajectories, `t_c` slopes are −0.7 to −0.8 (head) and −1.1 to −1.3 (tail) for Adam, and about −1 for SGD. At `K = Id`, `alpha = 2` the rarest Adam classes slow sharply (5–7 unlearned by `t = 1000` at `beta2 <= 0.95`, not a fixed point). |
| The E1 kink | **Present and quantitatively predicted at `Theta_0`**: within 0–15% of `kappa_beta (1-beta2)/(b F^2)` at `C = 8` and 32. The background prefactor `F` is essential at small `C` (factor 2–3 at `C = 8`). **Along trajectories** it moves with `beta2` as predicted (0.95/0.99 ratio 2.8–3.2 against 2.97; 0.9 and 0.95 coincide, as `kappa_beta` says), sits 1.5–2x above the frozen value at `K = Id` and 0.6–1.3x at `K = 0.1 Id`, and the rare-side Adam/SGD ratio matches its zero-parameter height to 0.9–1.3. |
| `beta2` | Below the kink, `t_c(0.95)/t_c(0.99)` = 1.72 against 1.795 predicted. `t_c(0.9)/t_c(0.95)` = 0.96–1.01 against 0.947, ruling out the naive `(1-beta2)` scaling (1.41). Shorter memory leaves more rare classes behind at `K = Id`. |
| Noise (`K = 0.1 Id`) | Frozen kinks are unchanged (exact rescaling). Along trajectories the break moves down to 0.6–1.3x the prediction, Adam's advantage keeps growing into the tail, and the rare-class slowdown disappears. The ODE needs larger `D` here (finite-D lag `~ r'^2/D`: 30% at `D = 1000`, 6% at 4000). |
| Bias | Little effect on risk. Adam's rare classes learn 10–35% slower with a bias at `K = Id`, because Adam keeps pushing rare biases below `log pi`. SGD is unaffected (4%). |

**Open questions and not done.**
- **Larger `C` and `beta2 = 0.999`**: `C = 128` (about 4 h of Adam ODE) is where E4's fixed point appears.
- **An Adam-bias ODE estimator**: the free stream + linear response design is in the plan file.
- **Why the trajectory kink sits about 2x above the frozen one at `K = Id`.** Candidate: the frequent classes'
  restoring term, which E1a found removes half of the tail drift by `t = 10`.
- **The late rise of Adam's risk at `C = 8`, `K = Id`.**
- **Learning rates tuned per `kappa`.** At `K = 0.1 Id` the fixed `eta` favours Adam, whose effective step does not
  shrink with the noise.

## Stage 3: next steps (2026-11-04/05)

The bias model is dropped from here on, since it added cost but no insight. Its Stage 1–2 figures were moved to
`analysis/bias_model_stages1-2/`. This stage adds:
- `beta2 = 0.999` at every `C`;
- `C = 128` (both `alpha`, both `kappa`);
- learning rates tuned per noise level;
- a diagnosis of the offset of the trajectory kink.

Commands: `figs/experiments/imbalance/logs/stage3.sh`. GPU time was about 2.8 h: 16:00–22:51 wall time, including two
probes that hung on compilation (below).

### Engineering

- **Compile time at `C = 128`.** One psi-route coefficient evaluation compiled for 496 s (12 s at `C = 32`); inside
  the ODE loop the first probe was still compiling after 25 min. The cause is the path layout.
  - With the default `chunk_elements = 2^25`, the 256 paths are split into 3 chunks, and XLA compiles the
    resulting short outer scan very slowly.
  - `imbalance.adam_plan` uses a single chunk (`2^27`) for `C > 32`. Compilation drops to 89 s, and a call costs
    0.31 s instead of 0.38 s.
  - `C <= 32` keeps the old layout, so its runs are bit-identical.
  - A persistent JAX compilation cache (`figs/experiments/imbalance/jaxcache`) lets runs that differ only in
    `kappa` or `alpha` reuse compiled code.
  - **Cost:** a `C = 128` Adam run to `T = 316` takes 300–600 Heun steps in 3.5–12 min, compilation included.
- **Learning-rate sets.** Runs now carry an `etaset` (`base` or `tuned`, from `eta_tuned.json`). A run whose `eta`
  exceeds the base value records from `t = 10^-k`, so that fast-learned classes are resolved. The first tuned runs
  recorded from `t = 0.01`; their head classes were learned before the first record time, which made the kink
  fits meaningless. They were rerun.
- **Margin drifts.** `frozen` also stores the drift of the logit margin `O_cc - mean_{k != c} O_kc`, taken from the
  full drift matrix `-(phi^T O + r^2 chi^T)`.
- **Diagnostic.** `offset` decomposes `t_A/t_S` (below).
- **Bug fix in `figures.py`.** `Path.with_suffix` truncated names such as `C_rates_m0_k0.1_a1` to `C_rates_m0_k0`,
  so the `alpha = 1` and `alpha = 2` panels for `K = 0.1 Id` overwrote each other, in Stages 1–2 as well. All
  figures were regenerated.
- **Recheck.** `recheck.py` reproduces all `t_c` and `R(T)` exactly and the ratio breaks within 0.7%; no flags.

### Learning rates per noise level

Tuning uses `C = 32`, `alpha = 1.5` and `beta2 = 0.99`, with criterion `R(t = 100)`.

| | ODE grid | ODE optimum | simulation at `D = 1000` (16 runs) | adopted (`tuned`) |
|---|---|---|---|---|
| `K = Id`, Adam | 0.1–3 | 0.3 | 0.3 (R(100) = 0.034) | 0.3 |
| `K = Id`, SGD | 0.3–30 | 3 | 3 (0.032) | 3 |
| `K = 0.1 Id`, Adam | 0.1–100 | 10 (30 rises late, 100 stalls at 2e-3) | 10 (5.8e-8) | 10 |
| `K = 0.1 Id`, SGD | 0.3–1000 | none: `R ∝ 1/eta` up to 1000 | 300 (5.0e-5); at 1000 the simulation leaves the ODE (`R(10)` 18x higher) | 300 |

- **At `K = Id`** tuning reproduces the E-series values in both the ODE and the simulation.
- **At `K = 0.1 Id` the ODE limit has no SGD optimum.** Separable, almost noiseless SGD is time-rescaled gradient
  flow. The finite-`D` process caps it, because the logit change per step on the next sample, about
  `eta kappa / sqrt(D)`, must stay small. Adam has a genuine optimum, because its normalised noise does not vanish
  with the gradient.
- **Tuning narrows Adam's low-noise advantage without removing it.**
  - At `t = 100` (`C = 32`, `alpha = 1.5`, simulation): `R_A/R_S` is 1/2600 at base rates and 1/860 at tuned
    rates.
  - The tuned ODE runs give `R_A/R_S ≈ 1/900` late for every `C <= 32` and `alpha`.
  - The exception is `beta2 = 0.999` at `eta = 10`: its risk rises late at `C = 32`, to 5–14 times SGD's by
    `t = 1000`.
- **Retuning `eta` does not move the kink.** The trajectory ratio breaks at `K = 0.1 Id` are 0.82–1.29 `kink_F`
  (tuned) against 0.86–1.26 (base) at `C = 32`, as theory says: the kink depends on `beta2`, not on `eta`.

### C = 128 (`T = 316` for Adam with `beta2 <= 0.99`, 1000 for 0.999; `10^4` for SGD)

| `kappa` | `alpha` | Adam `beta2` 0.9 / 0.95 / 0.99 / 0.999: classes learned, `R(T)` | SGD | `R_A/R_S` at t = 1 / 10 / 100 / 316 |
|---|---|---|---|---|
| 1 | 1 | **27 / 31 / 77 / 128**; 1.39 / 1.30 / 0.63 / 0.099 | 128; 0.043 (`10^4`) | 1.3 / 1.7 / 8.6 / 18 (0.9); 1.2 / 0.85 / 0.89 / 1.4 (0.999) |
| 1 | 2 | 22 / 22 / 27 / 57; 0.14 / 0.14 / 0.11 / 0.061 | 128; 0.011 | 1.6 / 1.3 / 1.8 / 2.8 (0.9); 1.6 / 0.84 / 0.97 / 1.4 (0.999) |
| 0.1 | 1 | all 128 | 128 | 0.85 / 0.51 / 7e-4 / 4e-4 (0.9); 0.67 / 0.018 / 3e-4 / 3e-4 (0.999) |
| 0.1 | 2 | 115 / 113 / 128 / 128 | 128 | 0.69 / 0.55 / 0.40 / 0.23 (0.9); 0.55 / 0.27 / 0.08 / 1.7e-3 (0.999) |

- **`alpha = 1`, `K = Id`: a genuine fixed point at short memory.**
  - At `beta2` 0.9 and 0.95 Adam stops with 27–31 of 128 classes learned.
  - The gaps are stationary to 5e-4 and 1.3e-3 (E4's criterion), and the late risk exponent is 0.00.
  - The risk is 18 times SGD's at `t = 316`.
  - Here the stuck classes are only moderately rare (`pi` from 1.4e-3 to about 7e-3). At `alpha = 2`, `C = 128`
    the stall is still moving (stationarity 0.11).
  - `beta2 = 0.99` learns 77 classes and is still moving. `beta2 = 0.999` learns all 128 by `t = 1000`.
  - Memory length, not imbalance alone, decides whether Adam gets stuck.
- **`alpha = 2`, `K = Id`: the rarest classes get worse first.**
  - 55–93 classes end above their initial loss.
  - `beta2 = 0.999` is the best Adam and still trails SGD at `t >= 100`.
- **`K = 0.1 Id`**: every class is learned except 13–15 of the rarest at `alpha = 2`, `beta2 <= 0.95`. Adam's risk
  ends 3–5000 times below SGD's.
- **Frozen kink at `C = 128`:** within 1–3% of `kappa_beta (1-beta2)/(b F^2)` for all four `beta2` (free slopes
  1.00 / 0.50). This reproduces E1a.
- **Validation at `D = 1000`, `C = 128`, `alpha = 2`:**
  - The time lag is 1.011 (Adam 0.99) and 1.007 (SGD).
  - At `beta2 = 0.999` the median lag is 0.98, but the minimum is 0.41: Adam's memory (1000 steps = 1 time unit) is
    not separated from the dynamics at `D = 1000`, so the warm-up dominates early.

### `beta2 = 0.999`

- **Frozen kink:** within 0–11% of `kink_F` at `C = 32` and 128. At `C = 8` the kink (0.023) lies at the rare
  edge of the class range.
- **Trajectory kink:** the ratio break is 1.79 `kink_F` (`C = 32`, `alpha = 2`) and 2.7 (`C = 128`, `alpha = 2`).
  The `alpha = 1` fits are not meaningful, because the rare side is barely in range.
- **`t_c(0.99)/t_c(0.999)` on the rare side** can be measured only at `K = 0.1 Id`, where the `beta2 = 0.99` tail is
  not stalled: 1.99 (`C = 32`) and 2.17 (`C = 128`), against 3.03 predicted. The window ratio is 0.97–1.04.

### Why the trajectory kink sits above the frozen one (`offset`, `F_offset.png`)

The half-loss time factorises as `t_c = O_cc(t_c) / vbar_c`, where `vbar_c` is the drift averaged along the path.
So `log(t_A/t_S)` splits into three parts:
- the **frozen ratio at `Theta_0`**;
- an **overlap factor**: how much `O_cc` Adam needs at `t_c` relative to SGD;
- a **path factor**: how far the averaged drift has fallen below its value at `Theta_0`, relative to SGD.

Medians at `C = 32`, `K = Id`, `alpha = 2` (log units):

| `beta2` | band | total | frozen | overlap | path |
|---|---|---|---|---|---|
| 0.9 | head (u > 3) | +1.51 | +2.05 | −0.48 | −0.07 |
| 0.9 | tail (u < 1/3) | +1.01 | +0.82 | −0.25 | +0.43 |
| 0.99 | head | +1.09 | +1.38 | −0.23 | −0.07 |
| 0.99 | tail | +0.72 | +0.28 | −0.05 | +0.44 |

- **The margin is equal at `t_c`.** `O_cc - mean O_kc` agrees between Adam and SGD to 1.00–1.03 in the head and
  near the kink. Adam reaches it with 20–40% less `O_cc`, because its normalisation amplifies the small background
  gradient `f_k = p_k ≈ 1/C` on the other columns. That pushes the competitors' overlaps `O_kc` down. SGD makes
  this push about `pi_c / C`; Adam makes it a significant fraction of the own-class push.
- **At `Theta_0`** this shows up directly. Adam's margin drift is 1.44–1.75 times its `O_cc` drift in the head and
  1.03–1.21 times in the tail; for SGD the factor is `1/(1 - 1/C)`.
- **Along the path at `K = Id`**, Adam's rare-class drift decays: the path factor is +0.25 to +0.44, i.e. 1.3–1.55
  times, and far more in the stall zone. This is the restoring term behind the stall. At `K = 0.1 Id` the path
  factor is −0.05 to −0.10 everywhere.
- **Both effects tilt `t_A/t_S`.** Adam gains in the head and loses in the tail, and the break moves to larger `pi`.
- **Replacing the `O_cc` drift by the margin drift** in the frozen prediction (the same broken-line fit, same
  classes), at `K = Id`:

  | | `C = 32` | `C = 128` |
  |---|---|---|
  | measured break / margin prediction | 1.07–1.40 | 1.11–1.61 |
  | measured break / `O_cc` prediction | 1.5–2.1 | 1.6–2.7 |

  The leftover factor is the path decay of the tail. At `K = 0.1 Id` the measured break sits below the margin
  prediction (0.53–0.95), because there is no path decay and the overlap factor favours Adam in the tail too.

## Summary (revised, 2026-11-05; C ∈ {8, 32, 128}, beta2 ∈ {0.9, 0.95, 0.99, 0.999}, b = 1)

| question | answer |
|---|---|
| Risk vs time | **`K = Id`:** the advantage shifts to SGD as imbalance and `C` grow. At `C = 8` Adam leads mid-training. At `C = 32` and 128 SGD is ahead at nearly all times, by up to 18 times at `C = 128`, `alpha = 1`, `beta2 <= 0.95`, where Adam is stuck. Longer memory (`beta2 = 0.999`) is the best Adam everywhere and roughly ties SGD up to `t ~ 100`. **`K = 0.1 Id`:** Adam wins by 3–4 decades late. With learning rates tuned for the noise level (Adam 10, SGD 300) Adam is still about 900 times better. |
| Per-class rates | At `Theta_0`, Adam's drift has slope ½ in `pi` above the kink and 1 below; SGD's has slope 1 exactly. Along trajectories Adam's `t_c` scales as `pi^-0.6` to `pi^-0.8` in the head and `pi^-1` to `pi^-1.3` in the tail; SGD's as about `pi^-1`. At `K = Id`, short memory and large `C` turn the slow tail into a stall: a fixed point at `C = 128`, `alpha = 1`, `beta2 <= 0.95`. |
| E1 kink | **At `Theta_0`:** within 0–15% of `kappa_beta (1-beta2)/(b F^2)` in all 12 (`C`, `beta2`) cases; `F` matters at small `C`. **Along trajectories:** it moves with `beta2` as predicted (0.95/0.99 ratio 2.8–3.2 against 2.97; 0.9 and 0.95 coincide) and does not depend on `eta`. At `K = Id` it sits 1.5–2.8 times above the `O_cc` prediction. **Most of that offset is the margin effect:** Adam's normalisation also pushes down competitor logits, which helps head classes most. Using the margin drift cuts the offset to 1.1–1.6; the remainder is the decay of the tail drift along the path (absent at `K = 0.1 Id`). |
| `beta2` | Rare side: 0.9/0.95 = 0.96–1.01 (0.947 predicted), 0.95/0.99 = 1.72–1.75 (1.795), 0.99/0.999 = 2.0–2.2 (3.03, measurable only at `K = 0.1 Id`). The main practical effect is the stall: shorter memory gets stuck. |
| Noise | `K = 0.1 Id` leaves the frozen kink unchanged, lowers the trajectory kink to at or below the frozen prediction, removes most of the stall, and makes Adam far better than SGD. At `D = 1000` it needs care: Adam has a finite-`D` lag of about `r'^2/D` (30% at `D = 1000`). |

**Open questions.**
- An analytic margin-drift kink: the `Theta_0` margin ratio needs `psi_k(1/C)` for the competitor columns.
- Why `t_c(0.99)/t_c(0.999)` falls short of 3.03 on the rare side at `K = 0.1 Id`.
- The fixed point at `C = 128`, `alpha = 1`, `beta2 <= 0.95`: its learned-class count against `beta2`, and whether
  `beta2 = 0.99` reaches one too.
- `D = 4000` simulations at `beta2 = 0.999`, where `D = 1000` is not in the high-dimensional regime.
