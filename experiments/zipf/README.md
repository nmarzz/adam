# Zipf-imbalanced multiclass GMMs: Adam scaling laws — log

Plan: `experiments/zipf_adam_plan.md`. Branch `zipf-scaling` (from `main` 04d8902).
Hardware: one H100 NVL, JAX 0.11.1 (`real/` venv), float64 throughout.
All commands are run from the repository root with `real/bin/python`; outputs
(`.npz`, `.pdf`, `.png`, logs) are in `figs/experiments/zipf/`.

## Model and fixed choices

* `C` classes, `pi_c ∝ c^{-alpha}`, means `x̄_c = r q_c` with `q_c` orthonormal
  (random `C`-dim subspace), `K = I`, softmax cross-entropy with `M = C` logits and
  no bias, `Theta_0 = 0.1 N(0, 1/D)`, `beta1 = 0.9`, stability `eps K_ii = eps`.
* **`r = 5`.** Monte Carlo with orthonormal means and `K = I` (200k samples):

  | C | alpha | Bayes error | no-bias rule `argmax_c q_c·x` |
  |---|---|---|---|
  | 8 | 1.2 / 2 | 0.09% / 0.07% | 0.12% / 0.14% |
  | 32 | 1.2 / 2 | 0.31% / 0.15% | 0.49% / 0.54% |
  | 128 | 1.2 / 2 | 0.73% / 0.22% | 1.22% / 1.17% |

  The no-bias rule is the best the model can represent; `r = 4` would give 9.6%
  and `r = 3` 35% at `C = 128`.

## Phase 0 — engineering

### What was added (all backward compatible)

| Where | What |
|---|---|
| `discounts.softmax_plan`, `softmax_coefficients` | Class-stratified path estimator for softmax with many classes (`phi`, `chi`, `sigma`; modes `adam`, `sgd`, `psgd`; derivative or Stein `phi`). Every coefficient is written as `sum_c pi_c E[contribution | sample-0 class = c]`; the class-conditional expectations pool ordinary window samples with **forced events** (one sample with a hand-set class) on extra paths, for every class with fewer than `min_events = 1000` ordinary events. A forced path holds one event per rare class (so a class never contaminates its own denominator), and forced samples are at most 2% of all samples. Analytic softmax Jacobian, contracted on the fly; paths processed in chunks; no `(paths, T, b, C, C)` tensor. |
| `discounts.softmax_standard_errors` | Batch-means standard errors of the diagonals of `chi`, `phi`, `sigma` (chunks are i.i.d.). |
| `discounts.softmax_moment_coefficients` | Path-free SGD / PSGD (`beta2 -> 1`) coefficients: these are single-sample expectations, so each class gets all `n = 2^15` common Gaussian draws, weighted by `pi_c` (exact stratification, rare classes as accurate as frequent ones). |
| `dynamics.isotropic_modes`, `run_mixture_ode(initial=...)` | Exact mode-summed state for `K = I` (`nu = 1`, Ito term `eta^2 sigma`). |
| `dynamics.run_mode_ode_logtime` | Heun with adaptive `dt = min(safety / (eta ||phi||_1 (1 + ||G||)), dt_max)`, shortened to land on arbitrary (log-spaced) record times; any coefficient callable. |
| `simulate.run_mixture_at(_many)` | Adam/SGD recorded at arbitrary step indices (log grid). |
| `problems.softmax_class_metrics` | Per-class `ell_c` and `err_c` from `(B, O)`, chunked over classes. |
| `experiments/zipf_scaling.py` | Subcommands `selftest timings precision tune validate rates timescales plot` (E2–E4 to come). |
| `tests/test_zipf_coefficients.py` | Self-tests (`real/bin/python tests/test_zipf_coefficients.py`). |

### Tests (`real/bin/python tests/test_zipf_coefficients.py`, 159 s): 13/13 pass

| Check | Result |
|---|---|
| New estimator vs `stationary_coefficients` (C = 5 mixture of `multiclass_batch.py`, b = 1 and 16, 6 seeds each, 75 entries) | Adam: rms z 1.12 / 1.19, max abs z 3.1 / 2.5; SGD: rms z 1.37 / 0.81 |
| Forced events vs ordinary events (C = 8, alpha = 1.5, all classes forced, 8 seeds) | class-mean diagonals of (phi, chi, sigma) agree within 0.2% (b = 1) and 0.5% (b = 16), all z < 1. A 16-seed study at b = 16 and forced fractions 2%, 0.5%, 0.01% found no bias (rel. diff ≤ 0.4%, z ≤ 1.05). |
| PSGD = SGD with preconditioner `(E f^2/b + eps)^{-1/2}` | 1.6e-14 |
| Path-free SGD/PSGD vs path-based | 0.12% / 0.19% |
| Collapsed `K = I` ODE vs mode-resolved ODE (D = 500) | 9.4e-16 relative |
| Log-time ODE step halving (safety 0.2 vs 0.05) | risk differs by 6.5e-5 |
| Log-time ODE (new estimator) vs `run_mixture_ode` (old estimator, fixed dt) | 0.67% max relative risk difference (Monte Carlo level) |
| `run_mixture_at` vs `run_mixture` at shared record steps | identical |

**Regression check.** `multiclass_batch.py batch --batches 4` reproduces the earlier
numbers exactly (Adam gap 0.0022, s.e. 0.0006; SGD 0.0005): ODE risk differs by 3e-16,
simulation is bit-identical. `multiclass_batch.py drift` (small) runs.

### Step size of the ODE (C = 32, alpha = 2, beta2 = 0.99, b = 1, eta = 0.1, t ≤ 1000; reference safety 0.1)

| safety | Heun steps | time | max abs risk diff | max rel diff of any `ell_c(t)` | max rel diff of `t_c` |
|---|---|---|---|---|---|
| 0.2 | 1599 | 91 s | 3.9e-5 | 1.7e-4 | 2.4e-5 |
| **0.5** | 664 | 55 s | 2.6e-4 | 1.3e-3 | 1.5e-4 |
| 1.0 | 354 | 42 s | 8.5e-4 | 4.6e-3 | 1.6e-4 |
| 2.0 | 199 | 36 s | 9.0e-4 | 1.3e-2 | 4.8e-4 |

Default for sweeps: **safety 0.5** (0.13% worst per-class error, below the Monte Carlo
error of the coefficients).

### Timings

One coefficient evaluation (`softmax_coefficients`, Adam, `alpha = 2`, 1000 forced
events per rare class; first block with `2^18` ordinary batches for every `b`, last
column with the adopted `max(2^14, 2^18/b)`):

| C | beta2 | b = 1 | b = 16 | b = 64 | b = 16 / 64, adopted |
|---|---|---|---|---|---|
| 8 | 0.95 / 0.99 / 0.999 | 2.5 / 2.8 / 3.0 ms | 38 / 33 / 33 ms | 123 / 128 / 132 ms | — |
| 32 | 0.95 / 0.99 / 0.999 | 11 / 19 / 69 ms | 94 / 94 / 159 ms | 334 / 396 / 367 ms | — |
| 128 | 0.95 / 0.99 / 0.999 | 95 / 113 / 296 ms | 236 / 443 / 2156 ms | 975 / 1544 / 8004 ms | 289 / 938 ms (0.99) |
| 256 | 0.95 / 0.99 | 356 / 389 ms | 599 / 1086 ms | 1667 / 3277 ms | 790 / 2121 ms (0.99) |

The cost is dominated by forced paths: each forced event needs ~`1/(1-beta2)` steps of
history free of other events of its class, times `b` samples per step. So `beta2 = 0.999`
with large `b` is expensive at large `C`; single (frozen-rate) evaluations are always
affordable, ODE trajectories are not.

Full ODE runs (`alpha = 2`, `beta2 = 0.99`, `eta = 0.1`, log grid 12/decade, safety 0.2):

| C | b | horizon | Adam | SGD (path-free) | PSGD (path-free) |
|---|---|---|---|---|---|
| 8 | 1 | 1000 | 19 s (986 steps) | — | — |
| 32 | 1 | 1000 | 100 s (1600) | 11 s (950) | — |
| 128 | 1 | 1000 | 728 s (2777) | 54 s (949) | 75 s (1410, 4/decade) |
| 128 | 16 | 300 | 1238 s (1930) | — | — |

With safety 0.5 the Adam runs need about 2.4x fewer steps (C = 128, b = 1: ~5 min).

### Rare-class precision (`precision` subcommand; batch means over ≥ 16 chunks)

Every class has at least 1006 pooled events in every configuration below.
Largest relative standard error over classes of the diagonals of (chi, phi, sigma), and
the value for the rarest class (`alpha = 2`, `eps = 1e-8`, at `Theta_0` and at a
mid-training state `O = 2 I`, `B = 0.3 I + 0.01`; values at `Theta_0` shown, the
mid-training values differ by at most 0.6 percentage points):

| C | beta2 | b | max over classes (chi / phi / sigma) | rarest class |
|---|---|---|---|---|
| 32 | 0.95 | 1 | 4.4 / 1.6 / 0.6 % | 3.0 / 0.3 / 0.4 % |
| 32 | 0.95 | 16 | 3.9 / 2.5 / 2.5 % | 3.6 / 1.4 / 1.8 % |
| 32 | 0.99 | 1 | 3.4 / 1.5 / 1.5 % | 2.2 / 0.7 / 1.1 % |
| 32 | 0.99 | 16 | 4.4 / 3.3 / 3.4 % | 3.2 / 2.2 / 2.4 % |
| 128 | 0.95 | 1 | 4.8 / 2.4 / 1.1 % | 2.9 / 0.2 / 0.1 % |
| 128 | 0.95 | 16 | 4.0 / 5.4 / 4.7 % | 2.6 / 1.3 / 1.4 % |
| 128 | 0.99 | 1 | 3.3 / 3.9 / 2.8 % | 2.1 / 0.3 / 0.4 % |
| 128 | 0.99 | 16 | 3.6 / 7.4 / 7.4 % | 2.7 / 2.0 / 2.8 % |

The ≥ 10^3-events criterion holds everywhere; the 3% criterion is met for the rarest
classes but not for every class (the maxima sit at classes just above the forcing
threshold, which get only ordinary events). For the frozen-rate figures (single
evaluations) I will use 4000 events per class; ODE vector fields use 1000.

## CHECKPOINT A — choices and proposed runs (waiting for go-ahead)

**Fixed:** `r = 5`, `beta1 = 0.9`, default `beta2 = 0.99`, `eps = 1e-8` (K-scaled),
Adam ODE safety 0.5, coefficient plans with `max(2^14, 2^18/b)` ordinary batches
(window 1024; 8192 for `beta2 = 0.999`) and 1000 forced events per class with fewer
than 1000 ordinary events (forced samples ≤ 2% of all samples), 16 runs per simulation.
`C ≤ 128` for anything that needs ODE trajectories (`C = 256` only for frozen rates or a
single E2 run); `beta2 = 0.999` ODE runs only with `b ≤ 4` or `C ≤ 32`.

**Proposed order and budget (≈ 4 h of GPU to Checkpoint B):**

1. `tune` (≈ 10 min): Adam `eta ∈ {0.03, 0.1, 0.3, 1}`, SGD `eta ∈ {0.3, 1, 3, 10}` on
   `C = 32`, `alpha = 1.5`, `b = 1`; criterion `R(t = 100)`; then fixed.
2. E0 `validate` (≈ 2.5 h): `C ∈ {8, 32, 128} × alpha ∈ {1.2, 1.5, 2} × b ∈ {1, 8, 32}`
   at `D = 4000` (horizons `t = 30 / 300 / 1000` for `C = 8 / 32 / 128`), plus
   `D ∈ {2000, 8000}` for `(C, alpha, b) ∈ {(32, 2, 1), (32, 2, 32), (128, 2, 8)}`.
   One ODE per `(C, alpha, b)`, started from the `D -> ∞` initial state `B_0 = 0.01 I`,
   `O_0 = 0` (so the finite-D fluctuation of `Theta_0` is part of the measured error).
3. E1 `rates` (≈ 25 min): frozen rates of `O_cc` at `C = 128`, `alpha = 2`, at `Theta_0`
   and at the Adam ODE state at `t = 10`, sweeping `beta2 ∈ {0.95, 0.99, 0.999}`,
   `b ∈ {1, 4, 16, 64}`, `eps ∈ {1e-8, 1e-4, 1e-2}` (at `b ∈ {1, 16}`), with PSGD and SGD;
   direct frozen-parameter Adam simulation for classes with `pi ≈ 0.07, 0.01, 0.001`
   (rarer classes would need ~10^9 steps: the normalized noise is O(1) per coordinate
   while the drift is O(pi_c)).
4. E1 `timescales` (≈ 45 min): `C = 64`, `alpha = 2`, `t ≤ 3000`, Adam / PSGD / SGD for
   `(beta2, b, eps) ∈ {(0.99, 1, 1e-8), (0.95, 1, 1e-8), (0.99, 16, 1e-8), (0.99, 1, 1e-4)}`,
   plus 8 simulated runs at `D = 4000` to `t = 300` for the first configuration.

Predicted kinks at `Theta_0` (see the note draft): `b pi = kappa_beta (1 - beta2)` with
`kappa_beta = (pi/2)((1 - beta1/sqrt(beta2))/(1 - beta1))^2 = 0.92, 1.43, 1.56` for
`beta2 = 0.95, 0.99, 0.999`; `pi = b eps`; and a softmax-background kink at
`pi_c (1-p_c)^2 ≈ sum_{k≠c} pi_k p_c^2` (`pi ≈ C^{-2}` at `Theta_0`), which the plan's
table does not include and which matters for the rarest classes when `alpha ≥ 2` or
`b ≳ (1 - beta2) C^2`.

---------------------------------------------------------------------------------
## Session 2 (2026-11-02): runs after Checkpoint A

Logs: `figs/experiments/zipf/run_to_B.log` (Oct 30) and `run_session2.log` (Nov 2). The
Oct 30 log stops at 19:07 during E0 at `C = 128` because the 10-hour slurm allocation
ended. Since commit 7df1486 the default estimator is the psi-route
(`discounts.softmax_psi_coefficients`), and every ODE run below uses it. The forced-event
estimator is only used in the precision study above.

**Operations (one H100, 2 CPU cores).**

* Run one GPU process at a time. Simulators and the ODE drive `lax.scan`/`while_loop` from the host, so each takes a
  full core. Two such processes on one GPU are time-sliced: a simulation that takes 2 min alone took 30 min next to
  another job.
* E2-E4 use ODE step factor `--safety 1`. On the `C = 64`, `alpha = 2` problem it is 1.8x faster than 0.5, and up to
  `t = 300` it changes the risk by at most 4.7e-4 (0.6%), any `ell_c` by at most 0.25%, `B_cc` by at most 0.09%, and
  no `t_c`.
* E2-E4 trajectories are cached by configuration in `figs/experiments/zipf/ode/`, with the full `B` and `O`, and start
  from the `D -> infinity` initial state. That start agrees with the finite-`D` `Theta_0` used by E1b to 0.1% in the
  risk at `t = 300`.

### Learning rates (`tune`, ODE only; `C = 32`, `alpha = 1.5`, `b = 1`, criterion `R(t = 100)`)

`real/bin/python experiments/zipf_scaling.py tune` (8 ODE runs, 6 min)

| | eta | R(t = 100) | R(t = 1000) |
|---|---|---|---|
| Adam | 0.03 / 0.1 / **0.3** / 1 | 0.218 / 0.068 / **0.034** / 0.056 | 0.023 / **0.016** / 0.017 / 0.040 |
| SGD | 0.3 / 1 / **3** / 10 | 0.183 / 0.065 / **0.032** / 0.033 | 0.028 / 0.016 / **0.013** / 0.026 |

Chosen and fixed (`eta.json`): Adam 0.3, SGD 3. At `t = 1000` Adam would prefer
`eta = 0.1`, which is the first sign of the late-time cost of a large Adam step studied in E4.

### E0 `validate`: ODE vs simulation

```
real/bin/python experiments/zipf_scaling.py validate          # Oct 30: C in {8, 32}, all alpha and b, D = 4000
real/bin/python experiments/zipf_scaling.py validate --val-classes 8 32 \
    --dsweep '[[8,2,8],[8,2,32],[32,2,1],[32,2,32]]' --val-dimensions 2000 8000 16000
```

Setup: one ODE per `(C, alpha, b)`, started from the `D -> infinity` initial state, and 16
Adam runs per `D` (`eta = 0.3`, `beta2 = 0.99`, `eps = 1e-8`). Horizons are `t = 30` for
`C = 8` and `t = 300` for `C = 32`. The ODEs took 2-514 s. The simulations took 7-170 s at
`D = 4000` and up to 1340 s at `D = 16000`. I stopped the `C = 32`, `b = 32`, `D = 16000`
case to free the GPU. Analysis script: `e0_analysis.py`; its numbers are quoted below.

There are two distinct effects:

1. **Warm-up transient (not an ODE error).** The simulator's Adam starts from `m = v = 0`
   with no bias correction. For the first ~`1/(1-beta2) = 100` steps it therefore takes
   sign-like steps, up to ~4x the stationary update size that the ODE assumes. The
   simulation runs ahead: its risk is below the ODE by 0.04-0.43 at `t = 0.01-0.1`, i.e.
   after 40-400 steps. The transient is larger for larger `b` and shrinks with `D` (log-log
   slopes -0.4 to -0.9), because a fixed number of steps occupies ODE time `~100/D`. The
   "sup mean gap" printed by `validate` (and in `run_to_B.log`) is this transient.
2. **After warm-up (`t >= 1`).** The simulation is slightly *behind* the ODE, and the gap
   decays like `1/D`:

| case | D = 2000 | D = 4000 | D = 8000 | D = 16000 | slope |
|---|---|---|---|---|---|
| C 8, alpha 2, b 8 | 0.0037 (5.4% of R, z 17) | 0.0018 (2.6%) | 0.0007 (1.0%) | 0.0004 (0.6%, z 5) | -1.09 |
| C 8, alpha 2, b 32 | 0.0024 (9.8%, z 48) | 0.0012 (4.8%) | 0.0006 (2.5%) | 0.0003 (1.3%, z 11) | -0.98 |
| C 32, alpha 2, b 1 | 0.0050 (0.9%, z 2.5) | 0.0022 (0.4%) | 0.0015 (0.2%) | 0.0017 (0.3%, z 2.7) | at the noise floor |
| C 32, alpha 2, b 32 | 0.0038 (3.0%, z 24) | 0.0019 (1.5%) | 0.0012 (0.9%, z 17) | stopped | -0.83 |

At `D = 4000`, the full grid (`C` in {8, 32}, `alpha` in {1.2, 1.5, 2}) gives 0.3-0.5% of
`R` with `|z| <= 2.4` at `b = 1`. It gives 0.3-3.2% at `b = 8` and 1.5-5.7% at `b = 32`.
The 0-1 error gap is at most 0.0018 in absolute terms. Final risks agree to four decimals.

**Per class** (worst relative gap of `ell_c` over `t >= 1`): the median over classes is
2-6% and is not larger for rare classes. Grouped by the expected number of class-`c`
events per unit time, `D pi_c`:

| D pi_c | 0-10 | 10-100 | 100-1000 | > 1000 |
|---|---|---|---|---|
| median relative gap | 4.5% | 3.8% | 3.5% | 3.3% |
| 90th percentile | 7.3% | 6.4% | 5.7% | 6.8% |

With only a handful of class-`c` events per unit ODE time, the rarest classes are as
accurate as the frequent ones. Their simulation spread is larger, so their z-scores are
smaller.

**Verdict (C <= 32).** Past the warm-up transient, the ODE is accurate to within MC error
at `b = 1`: 0.2-0.5% of the risk for `D >= 4000`. For larger batches the residual is a
finite-`D` bias that scales as `D^-1` and grows with `b` (about 2x from `b = 8` to
`b = 32`). It is below 1% of the risk at `D = 8000` (`C = 32`) and 1.3% at `D = 16000`
(`C = 8`, `b = 32`). The pass criterion holds at the largest `D` for `b <= 8`; at
`b = 32` it holds at `C = 32` and is narrowly missed at `C = 8`. Rare classes show no
extra error. Results at `C = 128` are pending (queue step 7).

### E1a `rates`: frozen per-class rates at C = 128 (H1a)

```
real/bin/python experiments/zipf_scaling.py rates --rates-check     # Oct 30, 52 min
```

**Setup.** `alpha = 2`. The rate of class `c` is the frozen drift `|dO_cc/dt|/eta = |(phi^T O + r^2 chi^T)_cc|`, from
psi-route coefficients averaged over 4 plan seeds. It is evaluated at `Theta_0` and at the Adam ODE state at `t = 10`
(`beta2 = 0.99`, `b = 1`, `eps = 1e-8`). There are ten Adam configs, with PSGD and SGD at the same states. The analysis
(`experiments/zipf/analysis/e1_rates/`, figure `figs/experiments/zipf/analysis/e1_rates/e1a_rates_summary.png`) splits
each rate into the forcing `r^2 pi_c E[psi_c(1 - p_c)]`, computed from the stored psi tables, and the restoring part
`-(phi^T O)_cc`. Slopes and kinks are fitted on the forcing, and every claim was recomputed independently by a second
script.

**Slopes and kinks at `Theta_0`.** Free fits give slope 0.99-1.00 on the rare side and 0.48-0.50 on the frequent side
in all ten configs, joined by a crossover about 2 decades wide. So a slope taken over a single decade near the kink
falls anywhere between 1/2 and 1. The kink `pi_b` is the intersection of the two asymptotes:

| config | kink `pi_b` | / `(1-beta2)/b` (plan) | / `kappa_beta (1-beta2)/b` | / `kappa_beta (1-beta2)/(b F^2)` |
|---|---|---|---|---|
| `beta2` 0.95 / 0.99 / 0.999, `b = 1` | 4.8e-2 / 1.5e-2 / 1.8e-3 | 0.95 / 1.48 / 1.75 | 1.03 / 1.03 / 1.12 | 1.02 / 1.01 / 1.01 |
| `b` = 4 / 16 / 64 (`beta2 = 0.99`) | 3.8e-3 / 1.1e-3 / 3.4e-4 | 1.52 / 1.69 / 2.18 | 1.06 / 1.18 / 1.52 | 1.00 / 1.00 / 1.00 |
| `b = 1`, `eps` 1e-4 / 1e-2 | 1.5e-2 / 3.0e-2 | 1.51 / 3.04 | 1.05 / 2.13 | 1.01 / 1.02 |
| `b = 16`, `eps` 1e-4 / 1e-2 | 3.1e-3 / 0.16 | 1.93 / 1.01 (vs `b eps`) | 1.93 / 1.01 | 1.01 / 0.99 |

`F` is a rare-side prefactor below 1. The rare-event law assumes that one class-`c` event dominates its own denominator.
The softmax background (`b/C^2` per step at `Theta_0`) and the eps floor compete with that event:

```
F(a) = sum_l rho^l g(a / (f^2 beta2^l)) / (sqrt(2/pi) sum_l rho^l),   rho = beta1/sqrt(beta2),
g(a) = E[xi^2/sqrt(xi^2 + a)],   f = 1 - 1/C,   a = (b^2 eps + b/C^2)/(1 - beta2).
```

* `F` = 0.995, 0.991, 0.947, 0.972, 0.922 and 0.810 for the `eps = 1e-8` configs in table order.
* It matches the measured rare-side prefactor at class 128 within 1% in nine configs (4% at `b = 64`, where class 128
  is not yet asymptotic).
* The window law is unchanged, so the kink moves to `kappa_beta (1-beta2)/(b F^2)`. That puts all ten measured kinks,
  including the eps kinks, within 2%.
* Divided by the window law `r^2 sqrt(b pi)`, the six `eps = 1e-8` curves collapse onto one function of
  `pi/kink_corrected`. They cross 0.5 at 0.33-0.36. Without `F` the crossings spread over 0.34-0.53.

**PSGD** has no `beta2` kink by construction, since `beta2 -> 1`:

* Its slope is 0.50-0.51 above `pi = 1e-3`, and it bends towards its own softmax-background kink, at local slope 3/4 at
  `pi = 6.1e-5` (predicted `1/((C-1)^2 - 1) = 6.2e-5`).
* At class 128 it overstates Adam's forcing by 22x, 12x and 4.2x for `beta2` = 0.95, 0.99 and 0.999, and by 6.2x, 3.3x
  and 1.9x for `b` = 4, 16 and 64.
* Adam itself has no background kink in this range. One would need `b >~ C^2 (1-beta2) = 164`.

**eps.** Only the larger edge of `pi_min = max(kappa (1-beta2)/b, b eps)` shows up, so an eps kink needs
`b > b* = sqrt((1-beta2)/eps)`:

* At `b = 1`, `eps = 1e-4` (`b* = 10`) the forcing is within 1% of the `eps = 1e-8` value at every class.
* At `b = b*` (`b = 1`, `eps = 1e-2`) the two edges merge.
* **The eps-regime rate does not depend on `b`.** At `b = 16`, `eps = 1e-2` the tail forcing is 0.997 of
  `r^2 pi (1 - 1/C)/sqrt(eps)`, and Adam/PSGD is 0.997-1.002 at every class. The plan's table (`sqrt(b) pi/sqrt(eps)`)
  is wrong by `sqrt(b)`. This follows from `eq:batch_chi`, because the batch sum and the `1/b` in `y` cancel once the
  denominator is the eps floor. Only this one config is fully eps-dominated.

**Where the own-class error sits on `psi_c`** (`|f| = 1 - 1/C`, at `Theta_0`, `beta2 = 0.99`, `b = 1`):

* Class 1 is on the linear part: `psi(|f|)/psi'(0) = 0.98`.
* Class 10 is in the crossover (0.21).
* Classes 50 and 128 are saturated (0.07, 98-99% of the plateau). For these `psi'(0) = 120-128 ~ C`, set by the
  background RMS `1/C`.
* At `b = 16`, `eps = 1e-2`, `psi'(0) = 1/sqrt(eps)` and `|f|` sits on the eps-linear segment.

This is exactly the plan's mechanism: the linear part gives the window, saturation gives the rare-event regime, and the
floor gives the eps regime.

**Direct simulation** (frozen Adam, 256 chains x 50 000 steps, `beta2 = 0.99`, `b = 1`):

| class (`pi`) | coefficients | simulation | difference (z) |
|---|---|---|---|
| 3 (0.068) | 5.896 +- 0.019 | 5.827 +- 0.004 | -1.2% (-3.7) |
| 8 (9.5e-3) | 1.516 +- 0.006 | 1.513 +- 0.003 | -0.2% (-0.5) |
| 25 (9.8e-4) | 0.1876 +- 0.0002 | 0.1887 +- 0.0022 | +0.6% (+0.5) |

The simulation also resolves the restoring term at class 25: the forcing alone would be 3 s.e. off. Classes rarer than
`1e-3` would need ~10^9 steps.

**Precision and finite-`D` scatter.** The 4-seed s.e. is at most 0.4% of the forcing (`b = 1`, `beta2 <= 0.99`) and 2%
at most anywhere. The class-3 gap shows that the total error can reach ~1%. The ragged rare tail of `rates_init.pdf` is
not MC noise. It is Adam's restoring term acting on the random initial overlaps `O_cc(Theta_0)` (s.d. 0.007 at
`D = 4000`):

* Over classes >= 64, rate minus forcing correlates with `O_cc` at r = -0.99.
* Adam's rare-class `phi_cc` is large (0.8-0.95 at `b = 1`), so relative to its forcing the restoring term is 14x that of
  SGD or PSGD.
* The log's "max rel s.e. 0.617" at `beta2 = 0.95` is class 125, whose net rate is nearly cancelled.

**Mid state (`t = 10`).**

* Classes 1-10 are learned, and the slope-1/2 window has disappeared.
* The tail forcing is unchanged (within 1% at `b = 1`), but the restoring term now removes much of the tail drift. The
  median over classes 25-128 is 68% at `beta2 = 0.95` and 50% at 0.99 (`b = 1`, `eps = 1e-8`), against 17%
  (`beta2 = 0.999`), 27% (`b = 4`), 11% (`b = 16`), 3% (`b = 64`), 12% (`eps = 1e-4`) and 1% for SGD and PSGD.
* This ordering is the order in which configs stall (E1b, E4).

**Verdict (H1a).** At `Theta_0` the frozen drift has H1's shape in every config: slopes 1/2 and 1, joined by a
~2-decade crossover. Checkpoint A's `kappa_beta (1-beta2)/b` places the `beta2` kink within 1.03-1.18x for `b <= 16`.
With the background/eps prefactor `F`, all ten kinks fall within 2%. PSGD misses the `beta2` kink. The eps edge appears
only when it is the larger edge, and the eps-regime rate is `r^2 pi/sqrt(eps)`, without the plan's `sqrt(b)`. All of
this is a statement about the initial drift: by `t = 10` the restoring term removes half of the tail drift at `b = 1`,
`eps = 1e-8`.

### E1b `timescales`: per-class learning times at C = 64 (H1b)

```
real/bin/python experiments/zipf_scaling.py timescales --timescale-check     # Oct 30, 79 min
```

**Setup.** `alpha = 2`, `t <= 3000`, starting from the finite-`D` `Theta_0`. Adam and PSGD (`eta = 0.3`) are run at
`(beta2, b, eps)` = (0.99, 1, 1e-8), (0.95, 1, 1e-8), (0.99, 16, 1e-8) and (0.99, 1, 1e-4). SGD (`eta = 3`) is run at
`b` = 1 and 16. `t_c` is the first time `ell_c <= ell_c(0)/2`. Adam's ODE took 400-2013 s, and PSGD and SGD took
14-37 s. Analysis: `experiments/zipf/analysis/e1_timescales/`, figure
`figs/experiments/zipf/analysis/e1_timescales/e1b_summary.png`.

**Exponents** (OLS slope of `log t_c` against `log pi_c`, by band):

| method | c1-3 | c4-6 | c7-14 | c15-26 | c27-64 | H1 |
|---|---|---|---|---|---|---|
| SGD `b` 1 | -1.02 | -1.04 | -1.08 | -1.14 | -1.21 | -1 |
| PSGD `b` 1 / 16 | -0.60 / -0.60 | -0.59 / -0.57 | -0.64 / -0.59 | -0.76 / -0.65 | -1.21 / -0.75 | -1/2 |
| Adam 0.99, `b` 1 | -0.66 | -0.78 | -1.08 | -1.88 | 3/38 learned | -1/2, then -1 below 1.4e-2 |
| Adam 0.95, `b` 1 | -0.72 | -0.93 | -1.28 | -2.68 (9/12) | none learned | -1/2, then -1 below 4.6e-2 |
| Adam 0.99, `b` 16 | -0.61 | -0.61 | -0.67 | -0.89 | -1.46 | -1/2, then -1 below 9e-4 |
| Adam 0.99, `b` 1, `eps` 1e-4 | -0.66 | -0.78 | -1.03 | -1.27 | -1.32 | as `b = 1` |

**Every method is steeper than its frozen-rate law.**

* SGD steepens from -1.02 to -1.21 although its frozen slope is 1.000.
* The PSGD window is -0.58 to -0.60 against a frozen slope of 0.50.
* Adam is a further 0.08-0.23 steeper than its frozen crossover.
* Part of the steepening of PSGD's tail at `C = 64` is static: the background kink sits at `1/C^2 = 2.4e-4`.

**Break vs kink.** This is a continuous broken-line fit with the break free, over the classes with `t_c <= 20`:

| config | slopes above / below | break / `kappa_beta (1-beta2)/b` |
|---|---|---|
| `beta2` 0.99, `b` 1 | -0.69 / -1.08 | 1.04 |
| `beta2` 0.99, `b` 1, `eps` 1e-4 | -0.69 / -1.05 | 1.06 |
| `beta2` 0.95, `b` 1 | -0.74 / -1.12 | 0.63 |
| `beta2` 0.99, `b` 16 | -0.63 / -1.13 | 1.99 |

* The break lies within a factor 2 of the predicted kink, but the factor drifts with the kink. It moves 2.0x between
  `beta2` = 0.95 and 0.99, where 3.2x is predicted, and 8.3x between `b` = 1 and 16, where 16x is predicted.
* Fixing the slopes at -1/2 and -1 moves the break to 2.5-5.5x and triples the rms.
* **Adam lags PSGD exactly where the kink says.** The ratio `t_c(Adam)/t_c(PSGD)` against `u = pi/kink` follows
  `max(1, u^-1/2)` (panel (e) of the figure). The lag reaches 1.5 at `u = 0.47-1.03`. At the head Adam is slightly
  faster than PSGD (0.84-0.95 at class 1).

**Matched-class laws hold.**

* Head, `sqrt(b)`: `t_c(b = 1)/(4 t_c(b = 16))` is 0.92-1.02 for classes 1-3.
* Rare end, linear in `b`: over classes 27-44, `t_c(b = 1, eps = 1e-4)/t_c(b = 16)` is 15.3-17.7, against 16.
* `beta2` dependence: for classes 11-14, `t_c(0.95)/t_c(0.99)` is 1.66-1.86. The rare-event law, which carries the
  factor `1 - beta1/sqrt(beta2)`, predicts 1.79.
* Collapse: the sub-kink classes with `t_c <= 20` collapse under `t pi` (s.d. of `log10(t_c pi_c)` 0.014-0.037). The
  window classes collapse better under `t sqrt(b pi)` than under `t pi` (0.11 vs 0.18 at `b = 1`), but not well. Within
  one run these collapses restate the band slopes.

**ODE vs simulation** (8 runs, `D = 4000`, `t <= 300`, first config):

* The classes learned are the same (1-27).
* For classes 5-27 the pooled `log(t_c sim / t_c ODE)` is -0.009 +- 0.006 (sum z^2 = 32 over 23 classes).
* Classes 1-4 are learned 4.3% earlier in simulation, a head start of 64-126 steps. This is E0's warm-up.
* For `t >= 1` the risk gap is at most 0.39% (|z| <= 1.9).

**Late-time stall at `b = 1`, `eps = 1e-8`.**

* Only 29/64 classes are learned by `t = 3000` at `beta2 = 0.99`, and 23/64 at 0.95. Every other run learns all 64.
* Classes 39-64 (34-64 at 0.95) end above their initial loss, by up to 1.33x.
* The tail leaves its `eps = 1e-4` control from `t ~ 14-25`, and the local `t_c` slope reaches -2 by `t ~ 59`.
* `B_cc(3000)` is about 700 in every column (1950 at `beta2 = 0.95`), against 5-27 for SGD and 12-37 for `eps = 1e-4`.

`B` alone does not single out the cause: PSGD carries as much rare-column `B` and learns every class. E4 resolves the
mechanism (below).

**Verdict (H1b).** H1 holds qualitatively and fails quantitatively. Its qualitative content is confirmed:

* `t_c` is shallow at the head and steep in the tail.
* The break lies within 0.6-2x of `kappa_beta (1-beta2)/b`.
* Adam lags PSGD below the kink as `u^-1/2`, while PSGD has no kink.
* The `sqrt(b)` head law and the linear-in-`b` rare law hold at matched classes.

Quantitatively, every method is steeper than its frozen law (by 0.1-0.2 in the exponent), and the break moves less than
predicted. For `b = 1`, `eps = 1e-8` the rare-class law is overtaken by a stall that frozen rates do not predict.

### Checkpoint B (written 2026-11-02, 13:05)

* **E0 passes** past the warm-up transient: <= 0.5% of the risk at `b = 1`, and an `O(1/D)` bias for larger `b`. The
  `C = 128` validation is still running (queue step 7).
* **H1 holds for the initial drift.** The kinks are within 2% after the background/eps correction, the eps-regime rate
  carries no `sqrt(b)`, and PSGD misses the `beta2` kink.
* **Along the trajectory H1 holds only qualitatively.** The stall at `b = 1`, `eps = 1e-8` is the failure that the plan
  says to diagnose before E2. The diagnosis (E4 below) is a *deterministic fixed point* of the drift, not a noise floor.
  By the plan's rule E2 would wait. The user asked to continue, so E2-E4 run with the stall in view and are interpreted
  through this mechanism.
