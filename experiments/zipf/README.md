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
