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

### Summary of findings (Nov 2)

| experiment | hypothesis | outcome |
|---|---|---|
| E0 validate | ODE accurate with many rare classes | **Passes** after Adam's warm-up transient. At `b = 1` the gap is 0.2-0.5% of the risk (also at `C = 128`). For larger `b` there is an `O(1/D)` bias. Rare classes are no less accurate. |
| E1a rates | H1, initial drift | **Holds**: slopes 1/2 and 1. All ten kinks fall within 2% of `kappa_beta (1-beta2)/(b F^2)`. The eps-regime rate has no `sqrt(b)`, contrary to the plan's table. |
| E1b timescales | H1 along the trajectory | **Qualitatively.** Every method is 0.1-0.2 steeper than its frozen law, and Adam lags PSGD as `max(1, u^-1/2)` below the kink. |
| E4 noise | H5, noise floor | **Refuted.** At `b = 1`, `eps <= 1e-6`, `beta2 = 0.99`, Adam stops at a *deterministic fixed point* with the rare classes unlearned (29/64), and direct simulation reproduces it. The cause is a normalization mismatch: the forcing is normalized by the class's own rare events, the curvature by the background. `B_cc` grows in a common mode the softmax ignores. `eps >= 1e-5`, `beta2 = 0.999` or `b >= 16` remove the stall; `b = 4` only moves it (43/64). |
| E2 exponents | H2, H3 | **Direction holds.** Inside the window Adam's risk exponent is 1.66-1.70x SGD's at `C = 64`, not 2x, because its frontier moves with `s ~ 0.6` against SGD's 1.05. H3's break appears at the window edge, at `t ~ 3/(eta r^2 sqrt(1-beta2))`. |
| E3 batch | H4 | **Supported for per-class targets.** The best `b` brackets `b* = 10` at `eps = 1e-4`. The best `b` grows as the target moves into the tail. The ODE is exactly batch-invariant for SGD with `eta ~ b` and PSGD with `eta ~ sqrt(b)`. |
| E5 bigram | transfer to the one-hot bigram model | **The `beta2`/batch kink transfers**: slope -1 below `kappa (1-beta2)/b`, shallower above, moving with `b`. **The stall does not**: one-hot inputs never mix classes in a coordinate's normalization. With the square loss Adam has a constant-step noise floor. |

**Operations (one H100, 2 CPU cores).**

* Run one GPU process at a time. Simulators and the ODE drive `lax.scan`/`while_loop` from the host, so each takes a
  full core. Two such processes on one GPU are time-sliced. The first E5 bigram configuration took 1780 s next to
  `simcheck`, against about 2 min estimated from a 28 s probe of `2^20` steps run alone, and `simcheck` itself slowed to
  32 min.
* The plan's 30-min limit per run was exceeded twice: the E1b Adam `b = 16` ODE took 2013 s, and the `C = 128` validation
  ODE 2177 s.
* E2-E4 use ODE step factor `--safety 1`. On the `C = 64`, `alpha = 2` problem it is 1.8x faster than 0.5, and up to
  `t = 300` it changes the risk by at most 4.7e-4 (0.03% of `R`), any `ell_c` by at most 0.25%, `B_cc` by at most 0.09%, and
  no `t_c`.
* E2-E4 trajectories are cached by configuration in `figs/experiments/zipf/ode/`, with the full `B` and `O`, and start
  from the `D -> infinity` initial state. That start agrees with the finite-`D` `Theta_0` used by E1b to 0.1% in the
  risk at `t = 300`.

### Learning rates (`tune`, ODE only; `C = 32`, `alpha = 1.5`, `b = 1`, criterion `R(t = 100)`)

`real/bin/python experiments/zipf_scaling.py tune` (8 ODE runs, 5 min)

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
case to free the GPU. Wall-clock: 46 min on Oct 30 for `C` in {8, 32} (the `C = 128` part was cut when the allocation
ended), and 53 min for the Nov 2 D-sweep. Analysis: `experiments/zipf/analysis/e0_validate/e0_analysis.py`, whose
numbers are quoted below. Figure: `figs/experiments/zipf/validate.pdf`.

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
The 0-1 error gap is at most 0.0018 in absolute terms. Final risks agree within 2e-4.

**Per class** (worst relative gap of `ell_c` over `t >= 1`): the median over classes is
2-6% and is not larger for rare classes. Grouped by the expected number of class-`c`
events per unit time, `D pi_c` (`C <= 32`; computed before the `C = 128` file existed):

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
extra error.

**`C = 128`** (`alpha = 2`, `b = 1`, `D = 4000`, `t <= 300`; `validate --val-classes 128 --val-alphas 2
--val-batches 1 8 --horizon-large 300`): the ODE took 2177 s (1049 steps) and the 16 runs 294 s. After warm-up the
risk gap is 0.0023 (0.21% of `R`, z = -0.9). The per-class worst relative gap has median 0.7%, and for the rarest
class it is 0.4% (z = 2.2), although that class gets `D pi_128 = 0.1` samples per unit time. I stopped the `b = 8`
case and its D-sweep because at 2.1 s per ODE step they did not fit the GPU budget.

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
* Among the configs later run on trajectories (E1b, E3), this ordering matches the stall: `beta2 = 0.95` (23
  learned) < 0.99 (29) < `b = 4` (43 at `b t = 3000`) < `b = 16`, `b = 64`, `eps = 1e-4` (all 64).

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

* Every run learns classes 1-27 by `t = 300`, as the ODE does; at least one run also learns class 28.
* For classes 5-27 the pooled `log(t_c sim / t_c ODE)` is -0.009 +- 0.006 (sum z^2 = 32 over 23 classes).
* Classes 1-4 are learned 4.3% earlier in simulation, a head start of 64-126 steps. This is E0's warm-up.
* For `t >= 1` the risk gap is at most 0.39% (|z| <= 1.9).

**Late-time stall at `b = 1`, `eps = 1e-8`.**

* Only 29/64 classes are learned by `t = 3000` at `beta2 = 0.99`, and 23/64 at 0.95. Every other run learns all 64.
* Classes 39-64 (34-64 at 0.95) end above their initial loss, by up to 1.33x.
* The tail leaves its `eps = 1e-4` control from `t ~ 14-25`, and the local `t_c` slope reaches -2 by `t ~ 59`.
* `B_cc(3000)` is about 700 in every column (1950 at `beta2 = 0.95`), against 5-27 for SGD and 12-37 for `eps = 1e-4`.

`B` alone does not single out the cause. PSGD's rare-column `B_cc` is about 1/5 of Adam's (median 156 vs 723 over
classes 39-64), but its part outside the mean span is 3x larger (`N_cc` 64 vs 19), and it learns every class. E4 resolves the
mechanism (below).

**Verdict (H1b).** H1 holds qualitatively and fails quantitatively. Its qualitative content is confirmed:

* `t_c` is shallow at the head and steep in the tail.
* The break lies within 0.6-2x of `kappa_beta (1-beta2)/b`.
* Adam lags PSGD below the kink as `u^-1/2`, while PSGD has no kink.
* The `sqrt(b)` head law and the linear-in-`b` rare law hold at matched classes.

Quantitatively, every method is steeper than its frozen law (by 0.1-0.2 in the exponent), and the break moves less than
predicted. For `b = 1`, `eps = 1e-8` the rare-class law is overtaken by a stall that the frozen rates at `Theta_0` do not
predict. The `t = 10` frozen rates (E1a) already show the restoring term cancelling 50-68% of the tail drift.

### Checkpoint B (written 2026-11-02, 13:05; the E4 diagnosis and the `C = 128` result were added later)

* **E0 passes** past the warm-up transient for `b <= 8`: <= 0.5% of the risk at `b = 1` and `D >= 4000`. At `b = 32`
  an `O(1/D)` bias remains, which narrowly misses the criterion at `C = 8` (1.3% at `D = 16000`). At
  `C = 128` (`b = 1`) the gap is 0.21% of the risk.
* **H1 holds for the initial drift.** The kinks are within 2% after the background/eps correction, the eps-regime rate
  carries no `sqrt(b)`, and PSGD misses the `beta2` kink.
* **Along the trajectory H1 holds only qualitatively.** The stall at `b = 1`, `eps = 1e-8` is the failure that the plan
  says to diagnose before E2. The diagnosis (E4 below) is a *deterministic fixed point* of the drift, not a noise floor.
  By the plan's rule E2 would wait. The user asked to continue, so E2-E4 run with the stall in view and are interpreted
  through this mechanism.

### E4 `noise`: eta sweep and the rare-class stall (H5)

```
real/bin/python experiments/zipf_scaling.py noise --safety 1 --noise-eps 1e-8                          # 40 min
real/bin/python experiments/zipf_scaling.py simcheck                                                    # 32 min
real/bin/python experiments/zipf_scaling.py noise --safety 1 --noise-eps 1e-4 --noise-modes psgd adam   # 15 min
```

**Setup.** `C = 64`, `alpha = 2`, `b = 1`, `beta2 = 0.99`. Adam and PSGD: `eta` in {0.03, 0.1, 0.3, 1} to
`eta T = 900`, `eps` in {1e-8, 1e-4}. SGD: `eta` in {0.3, 1, 3, 10} to `eta T = 9000`. `simcheck`: 8 direct Adam runs
(`D = 2000`, `eta = 0.3`, `eps = 1e-8`, one shared finite-`D` `Theta_0`) to `t = 3000`. Analysis:
`experiments/zipf/analysis/e4_noise/e4_analysis.py`, rerun after the `eps = 1e-4` runs. The equal-`t` and simulation
tables and the coefficient checks come from the verification agents' scripts. These used numpy plus one 80 s GPU
job recomputing psi-route and plain-path coefficients at the end state, and they are kept outside the repository. Figures:
`figs/experiments/zipf/noise.pdf`, `figs/experiments/zipf/analysis/e4_noise/e4_mechanism.png`.

**Runs** at `T`; cells are `eta` = 0.03 / 0.1 / 0.3 / 1 (SGD 0.3 / 1 / 3 / 10). Learned: `ell_c <= ell_c(0)/2` at some
record time. `gap_c = O_cc - max_{k != c} O_kc`. `N = B - O O^T/r^2` is the part of `B` outside the span of the means.
Stationarity: largest change of the gaps `O_cc - O_kc` over `[T/sqrt(10), T]`, relative to the largest gap. `s`: slope
of `log gap_c` against `log pi_c` (classes 16-64). `R(T)`, learned and `max B_cc` match `run_session2.log`.

| run | R(T) | learned | max B_cc | N_cc/B_cc (%) | gap_64 | stationarity | s |
|---|---|---|---|---|---|---|---|
| Adam, `eps` 1e-8 | 0.0664 / 0.0683 / 0.0741 / 0.0775 | 28 / 28 / 29 / 35 | 1158 / 1039 / 726 / 176 | 0.25 / 0.85 / 2.7 / 17 | 0.454 / 0.475 / 0.579 / 5.10 | 7.5e-5 / 7.9e-5 / 2.9e-4 / 0.14 | 1.14 / 1.14 / 1.12 / 0.50 |
| Adam, `eps` 1e-4 | 0.0079 / 0.0087 / 0.0116 / 0.0308 | 64 | 27.9 / 30.0 / 37.5 / 92.8 | 1.1 / 3.6 / 10 / 29 | 10.4 / 10.6 / 11.3 / 15.9 | 0.18 / 0.18 / 0.20 / 0.26 | 0.13-0.16 |
| PSGD, `eps` 1e-8 | 0.0100 / 0.0183 / 0.0418 / 0.1328 | 64 / 64 / 64 / 29 | 83 / 156 / 340 / 936 | 10 / 21 / 32 / 45 | 24.8 / 32.6 / 44.6 / 66.4 | 0.36 / 0.40 / 0.41 / 0.42 | 0.08-0.14 |
| PSGD, `eps` 1e-4 | 0.0079 / 0.0128 / 0.0323 / 0.1138 | 64 / 64 / 64 / 33 | 49 / 82 / 229 / 764 | 5.4 / 15 / 29 / 43 | 19.1 / 23.8 / 36.2 / 60.7 | 0.28 / 0.33 / 0.39 / 0.41 | 0.08-0.12 |
| SGD | 0.0079 / 0.0083 / 0.0100 / 0.0215 | 64 | 21.4 / 22.7 / 27.0 / 58.7 | 1.0 / 3.1 / 9.0 / 26 | 10.3 / 10.5 / 11.1 / 14.3 | 0.21 / 0.22 / 0.23 / 0.26 | 0.12-0.13 |

* **Adam with `eps = 1e-8` and `eta <= 0.3` stops at a fixed point.** It learns 28-29 classes at every `eta` and never
  loses one; the gaps are stationary to 3e-4; 21-26 rare classes (39-64 at `eta = 0.3`) end above their initial loss,
  by up to 1.34x. The margins grow mildly with `eta` (`gap_32` = 2.44 / 2.61 / 3.33 at `eta` = 0.03 / 0.1 / 0.3) and
  scale as `pi_c^1.12-1.14` (local exponent 0.9-1.0 over classes 16-39, 1.14-1.17 over 40-64).
* **`eps = 1e-4` removes the stall** at every `eta`: all 64 classes are learned and the margins are still growing. The
  floor sets the denominator (`psi_c'(0)` = 91-99 against `1/sqrt(eps)` = 100), so `phi_cc` follows the background
  scale (0.0005-0.0023 at `T`, decaying like `t^-0.67`). PSGD and SGD have no fixed point either.

**Equal `t`** (Adam, `eps = 1e-8`, `eta` = 0.03 / 0.1 / 0.3 / 1; head = classes 1-16, tail = 17-64, prior-weighted):

| t | R(t) | head loss | tail loss | learned |
|---|---|---|---|---|
| 30 | 0.445 / 0.217 / **0.139** / 0.147 | 0.340 / 0.109 / **0.038** / 0.039 | 4.13 / 4.01 / **3.68** / 3.94 | 6 / 11 / 17 / **20** |
| 100 | 0.213 / 0.122 / **0.094** / 0.122 | 0.106 / 0.028 / **0.017** / 0.032 | 3.98 / 3.42 / **2.79** / 3.28 | 11 / 18 / 23 / **25** |
| 300 | 0.124 / 0.083 / **0.077** / 0.099 | 0.030 / **0.013** / 0.014 / 0.028 | 3.44 / 2.55 / **2.31** / 2.59 | 17 / 24 / 27 / **30** |
| 900 | 0.083 / **0.070** / 0.074 / 0.078 | 0.013 / **0.011** / 0.013 / 0.026 | 2.54 / 2.14 / 2.22 / **1.89** | 24 / 28 / 29 / **35** |

`eta = 0.3` is best for the risk and the tail loss up to `t = 300`. At `t = 900` `eta = 0.1` wins the head and the
risk, and `eta = 1` the tail loss. `eta = 1` learns the most classes at every `t` but raises `ell_64` the most (to
1.6-2.1x its initial value). Best risks at `t = 900`: PSGD 0.0134-0.0149, SGD 0.0146, Adam 0.0186 (`eps = 1e-4`).

**Simulation** (mean +- s.e. over the 8 runs, against the `eta = 0.3` ODE interpolated in `log t`):

| t | R sim / ODE | learned, running | learned at `t` | gap_64 sim / ODE | mean-competitor gap_64 |
|---|---|---|---|---|---|
| 303 | 0.07714 +- 0.00020 / 0.07725 | 27.4 / 27 | 27.1 / 27 | 0.32 +- 0.06 / 0.54 | 0.62 +- 0.06 / 0.61 |
| 953 | 0.07417 +- 0.00024 / 0.07414 | 29.1 / 29 | 28.4 / 29 | 0.43 +- 0.05 / 0.58 | 0.75 +- 0.05 / 0.66 |
| 3000 | 0.07438 +- 0.00017 / 0.07410 | 29.75 +- 0.16 / 29 | 28.4 +- 0.3 / 29 | 0.34 +- 0.08 / 0.58 | 0.65 +- 0.06 / 0.66 |

Real Adam stalls the same way. The risk agrees within 0.6% (`|z| <= 1.9`) for `t = 10-3000`, except at `t = 403` (+0.68%, z 3.3), and is flat late
(`R(3000) - R(953)` = +0.00021 +- 0.00037). `ell_64(3000)` = 5.565 +- 0.059 vs 5.563, rare margins do not move after
`t = 953`, and `B` grows as in the ODE (common-mode share 0.9959 vs 0.9957). The min-gap is 0.2-0.3 low in every
class: the max over about 63 nearly tied competitors, each with a steady run-to-run s.d. of 0.21-0.235, is biased up.

**Mechanism: a fixed point of the shift-invariant drift.** The softmax ignores a common logit shift, so `1^T phi = 0`
and `(P O, P B P)`, `P = I - 11^T/C`, is closed. At the `eta <= 0.3` end states, psi-route coefficients recomputed
with the ODE's own plan give `|P dO/dt|` = 0.9-1.9e-6 and `|P dB/dt P|` about 3e-6 of the forcing
`eta |P r^2 chi^T|`. Other plans or an independent numpy `phi` leave a mixed-sign 3% rms (MC level). Stationarity
reads `phi^T o_c + r^2 chi_{c.}^T = lambda_c 1` (`o_c` = column `c` of `O`). The columns of `phi` sum to 0 and
`phi_sc <= 0` for `s != c`, so row `c` fixes a weighted margin,
`mbar_c = sum_{s != c} (|phi_sc|/phi_cc)(O_cc - O_sc) = (r^2 |chi_cc| + lambda_c)/phi_cc`. At `eta = 0.3`, for `c` =
32 / 48 / 64, `gap_c` = 3.33 / 1.175 / 0.579, `mbar_c` = 3.42 / 1.225 / 0.608 and the right-hand side 3.64 / 1.211 /
0.612. `lambda_c` (0.0066-0.024) is the common-mode drift of column `c`, predicted to 3e-5; without it the rare
margins come out 10-21% low.

**Why `phi_cc` stays finite.** Adam normalizes the forcing by the own event and the restoring term by the background.
For classes >= 32, `r^2 chi_cc/pi_c` is 97-99% of `-r^2 c_beta sqrt(2/pi)/(1 - beta1/sqrt(beta2)) = -208.9`, the
sign-like plateau. The curvature is `phi_cc = kappa E_bg[p_c(1-p_c)]/RMS_bg(p_c)`, `kappa = 0.44-0.94`, with `E_bg`
over other classes' inputs: own events carry 200-400x the background in `E f_c^2`, yet `psi_c'(0)` = 1.7-1.9 /
`RMS_bg`. So `phi_cc` ignores the scale of the background, not its shape. It fell 11-20x as the background shrank and
grew a heavy tail, then stopped with `P B P` at 0.034-0.081; `eps = 1e-4` would cut it 5-8x. Shrinking `E_bg[p_64]`
55x at the `eta = 0.3` state (logit 64 lowered by 4 on other means, `eps -> 0`) moves Adam's balance margin
0.493 -> 0.532, but SGD's and PSGD's (equal on the diagonal) 47.7 -> 1399: they normalize forcing and restoring alike.

**Noise.** Exactly, `dN/dt = -eta Sym(N phi) + eta^2 sigma` (the `O chi` terms of `dB/dt` and `d(O O^T)/dt` cancel),
and `dO/dt` has no `sigma` term: noise reaches the margins only through the coefficients. At the fixed point the
differential noise is in Lyapunov balance (residual 6e-6). 99.6-99.9% of `tr B/C` is the common mode driven by
`lambda_c`, which Adam's per-output normalization does not conserve (SGD conserves it), so `B_cc` grows like
`t^1.8-1.95`. At `eta = 1` the differential variance keeps growing (`Var(q_64 - q_1)` 16.6 -> 75.3 over
`eta t` = 9-900, against plateaus of 11.9-17.4 at `eta <= 0.3`), so `phi_cc` keeps falling and the margins follow.

**Verdict (H5).** Refuted as stated. H5 predicted a noise floor: rare-class `B_cc` plateauing near `eta/sqrt(b pi_c)`
and an `eta` limited by the tail. `B_cc` never plateaus (its local growth exponent at `T` is 1.85-1.96, in a common mode the softmax ignores), and
at `eta <= 0.3` its diffusion-driven part is 0.25-2.7%. The tail does stall at `b = 1`, `eps -> 0`, but at a
deterministic fixed point of the drift (28-29 classes at every `eta <= 0.3`), set by the mismatch between own-event and
background normalization; `eps = 1e-4` removes it, and PSGD (`eta <= 0.3`) and SGD learn all classes. At fixed `t` the
head sets the optimal `eta` and the tail would take a larger one. At `t = 900` Adam (`eps = 1e-8`) has 5x SGD's risk.

**Dropped claims** (refuted, superseded above, or not decidable from the stored data):

* Margins `~ pi_c^1.07-1.14` (1.07 not reproduced); `B_cc` "grows to 700-1160" (no plateau); `N` "1-3% of `B_cc`"
  (0.25-2.7%); `phi_cc` "stays O(1)" (it falls 11-20x first); stationarity `phi^T o_c = -r^2 chi_{c.}^T` (only up to
  the common mode `lambda_c 1`); "more noise (`eta = 1`) moves the fixed point" (none is reached by `eta T = 900`).
* PSGD's `phi_cc` "decays like `1/t` because own events dominate `E f_c^2`": it decays like `t^-1/2`, and own events
  are 41-69% of `E f_c^2` at its end state; what separates it from Adam is the shared normalization.
* The rare-event condition `b pi_c << 1 - beta2` needs a log factor (own events dominate `S_c` for 850-980 steps).
* "The fixed point holds with every estimator": the forced-event estimator is biased here (`phi_cc` 10-16% and
  `sigma_cc` 27% low against 8.4M plain-path samples; spurious +10-12% margin growth). Do not use it at late times.
* Simulation "29.8 vs 29 learned" (a running count, inflated by threshold crossings; 28.4 +- 0.3 vs 29 at fixed `t`)
  and "`c = 64` margin 0.32-0.43 vs 0.58" (a finite-`D` bias of the min-gap; the mean-competitor margin agrees).
* Not decidable: whether a larger `b` keeps a fixed point. The E2 `b = 16` run learns all 64 classes by `t = 1000`,
  `gap_64` is 9.5, and its gaps still move.
* Partly decidable: whether the stall exists at other `C`. At `C = 128` (E0, `alpha = 2`, `b = 1`, `t <= 300`) the same
  signature appears in the ODE and the simulation alike: 27/128 classes learned, and classes 41-128 above their initial
  loss by up to 1.28x. Whether it is a fixed point there needs `t >> 300`.
* PSGD at `eta = 1` also loses its tail, without a fixed point: 29/64 learned, 16 classes above their initial loss by up
  to 1.5x, `R = 0.133`, and `N_cc/B_cc` up to 64% (33/64 learned at `eps = 1e-4`). This is the only E4 case where a
  noise-driven tail limit (H5) is plausible. It was not analysed further.

#### E4 diagnostics: where the fixed point ends

```
real/bin/python experiments/zipf_scaling.py noise --safety 1 --noise-eps 1e-6 1e-5 --noise-adam-etas 0.3 --noise-modes adam   # 11 min
real/bin/python experiments/zipf_scaling.py noise --safety 1 --noise-eps 1e-8 --noise-adam-etas 3 --noise-modes psgd adam      # 2 min
real/bin/python experiments/zipf_scaling.py exponents --safety 1 --exponent-alphas 2 --exponent-batches 4 --exponent-modes adam --exponent-horizon 3000   # 14 min
real/bin/python experiments/zipf_scaling.py noise --safety 1 --beta2 0.999 --noise-eps 1e-8 --noise-adam-etas 0.3 --noise-modes adam   # 4 min
```

Every run below is Adam, `C = 64`, `alpha = 2`, and changes one setting from the reference run (`b = 1`,
`beta2 = 0.99`, `eps = 1e-8`, `eta = 0.3`, `T = 3000`). Stationarity is defined as in the E4 table: the largest change
of the logit gaps over the last half-decade, relative to the largest gap. "Above initial" counts classes whose
`ell_c(T)` exceeds `ell_c(0)`. `s` is the slope of `log gap_c` against `log pi_c` over classes 16-64.

| change | R(T) | learned | above initial | stationarity | gap_64 | s | phi_64(T) |
|---|---|---|---|---|---|---|---|
| none (reference) | 0.0741 | 29 | 26 | 2.9e-4 | 0.58 | 1.12 | 0.062 |
| `eps = 1e-6` | 0.0622 | 32 | 20 | 2.0e-3 | 0.95 | 0.91 | 0.038 |
| `eps = 1e-5` | 0.0238 | 58 | 0 | 0.11 | 4.8 | 0.33 | 0.0063 |
| `eps = 1e-4` | 0.0116 | 64 | 0 | 0.20 | 11.3 | 0.14 | 0.0011 |
| `b = 4` | 0.0380 | 43 | 8 | 6.3e-5 | 1.54 | 0.73 | 0.087 |
| `beta2 = 0.999` | 0.0228 | 64 | 0 | 0.24 | 15.1 | 0.17 | 0.0039 |
| `eta = 1` (`T = 900`) | 0.0775 | 35 | 18 | 0.14 | 5.1 | 0.50 | 0.0060 |
| `eta = 3` (`T = 300`) | 0.174 | 24 | 26 | 0.31 | 22.8 | 0.30 | 0.0002 |
| PSGD, `eta = 3` (`T = 300`) | 0.423 | 13 | 44 | 0.44 | 96.6 | 0.19 | 0.0008 |

* **`eps`.** The stall fades between `eps = 1e-6` (still nearly stationary, 32 learned) and `1e-5` (no fixed point, 58
  learned, no class above its initial loss). The floor matters once the background second moment of a rare output falls
  below about `eps`, which matches the verification estimate that the reference fixed point sits about 7x above the
  `eps = 1e-8` floor in background scale.
* **`b = 4` moves the fixed point without removing it.** The `t = 750` (E3) and `t = 3000` runs end at the same state:
  43 learned, `R = 0.0380`, gaps stationary to 6e-5. Its rare margins are larger (`gap_64` 1.54 vs 0.58) and flatter in
  `pi` (`s` = 0.73). Whether `b = 16`, which learns all 64 classes, also sits at a fixed point is being checked with a
  `t = 3000` run.
* **`beta2 = 0.999` removes it.** All 64 classes are learned, although its kink (`kappa (1-beta2) = 1.6e-3`) still puts
  classes 21-64 in the rare-event regime. A 1000-step memory keeps a rare class's own events in its denominator much of
  the time, so the curvature is no longer normalized by the background alone. The fixed point needs a memory that is
  short compared with the time between a class's own events.
* **More noise helps only up to a point.** `eta = 1` learns 35 classes and is still moving at `T = 900`. At `eta = 3`
  the diffusion dominates: the margins are large but so is the differential logit variance, and the risk is 0.17 for
  Adam (24 learned) and 0.42 for PSGD (13 learned).

### E2 `exponents`: risk exponents (H2, H3)

```
real/bin/python experiments/zipf_scaling.py exponents --safety 1     # 37 min
```

**Setup.** `C = 64`, `alpha` in {1.25, 1.5, 2, 3}, `b` in {1, 16}, Adam and PSGD at `eta = 0.3` and SGD at `eta = 3`.
`beta2 = 0.99`, `eps = 1e-8`, `t <= 1000` with 16 records per decade. Analysis:
`experiments/zipf/analysis/e2_exponents/e2_analysis.py`. Figure: `figs/experiments/zipf/exponents.pdf`. Its middle
panels draw the plan's asymptotic exponents and omit PSGD. The claims below were recomputed independently, and the
corrections are included.

Not done from the plan:

* `C > 64`. An Adam ODE step costs 2.1 s at `C = 128`, so a `C = 128` exponent run would take hours.
* The exponent of the 0-1 error.
* Simulation spot-checks other than E4's.

**Frontier model at finite `C`.** If `t_c ~ pi_c^-s`, the frontier (the number of learned classes) grows as
`c*(t) ~ t^(1/(alpha s))`. For `alpha <= 2` the risk is mostly the unlearned tail, `R ~ sum_{c > c*} pi_c ell_c(0)`
(51-87% of `R` in the fit window); at `alpha = 3` learned classes carry up to 60% of it. At `C = 64` the tail is
truncated and the first classes are discrete, so the local exponent is `(-dlog tail/dlog c*)/(alpha s)`, with tail
`int_{c*+1/2}^{C+1/2} k^-alpha dk`, instead of the plan's `2(alpha-1)/alpha` (`s = 1/2`) or `(alpha-1)/alpha`
(`s = 1`). Truncation dominates at `alpha <= 1.5`. At `alpha = 3` discreteness accounts for the change, since
truncation alone would give 1.35. Averaged over `c*` in [2, 16]:

| alpha | 1.25 | 1.5 | 2 | 3 |
|---|---|---|---|---|
| plan, `s = 1/2` / `s = 1` (`C -> infinity`) | 0.40 / 0.20 | 0.67 / 0.33 | 1.00 / 0.50 | 1.33 / 0.67 |
| finite `C = 64`, `s = 1/2` / `s = 1` | 0.87 / 0.44 | 0.92 / 0.46 | 1.03 / 0.52 | 1.23 / 0.62 |

A staircase model fitted the same way as the measurement gives values 0.02-0.08 higher, so the comparison carries that
much discretization ambiguity.

**Measured.** Exponents are least-squares fits over the times where `2 <= c* <= 16`. The effective `s` comes from
per-class fits of `log t_c` against `log pi_c` over classes 2-16. The slope of the integer frontier biases `s` low by
about 8%: 0.95-0.98 for SGD, 0.54-0.58 for Adam.

| alpha | exponent SGD (`b` 1 / 16) | Adam `b` 16 | PSGD `b` 16 | Adam/SGD (PSGD/SGD) | `s`: SGD / Adam `b` 16 / PSGD `b` 16 |
|---|---|---|---|---|---|
| 1.25 | 0.40 / 0.41 | 0.69 | 0.71 | 1.70 (1.76) | 1.04-1.06 / 0.59 / 0.57 |
| 1.5 | 0.43 / 0.44 | 0.74 | 0.77 | 1.70 (1.75) | 1.04-1.06 / 0.60 / 0.57 |
| 2 | 0.49 / 0.50 | 0.83 | 0.89 | 1.66 (1.77) | 1.04-1.06 / 0.63 / 0.58 |
| 3 | 0.59 / 0.60 | 0.84 | 1.01 | 1.41 (1.69) | 1.04-1.06 / 0.77 / 0.61 |

* **SGD** lies 0.01-0.04 below the finite-`C` model with `s = 1` at every `alpha`, and its per-class `s` is 1.04-1.06.
  Its local exponent is flat (within 5%) over about two decades at `alpha >= 2`: 0.47-0.51 at `alpha = 2`, the plan's
  0.5.
* **Adam and PSGD inside the window.** At `b = 16` the `beta2` window holds the first 64, 60, 26 and 9 classes.
  * For `alpha <= 2` Adam's frontier moves with `s = 0.59-0.63`, rising from about 0.58 on classes 2-6 to 0.62-0.68
    on classes 7-16.
  * The risk exponent is 1.66-1.70x SGD's for Adam and 1.75-1.77x for PSGD, against the plan's doubling. Finite `C`
    shifts both exponents but not their ratio. The ratio is the ratio of frontier speeds (Adam `s ~ 0.6`, SGD
    `~ 1.05`), and the residual loss of learned classes changes it by at most 7%.
  * The ratios move by about +-0.05 with the fit window or the learned threshold.
  * At `alpha = 3` only 9 classes are in the window: Adam's `s` is 0.77 and the ratio is 1.4 (1.28-1.62 depending on
    the window).
  * No Adam local exponent is flat (within 5%) over more than about 1.2 decades.
* **Broken power law (H3).** Fitting Adam's exponent separately for `c*` in [2, 6] and [6, 16]:
  * At `b = 1`, `alpha = 2` (window of 6 classes) it drops from 0.71 to 0.52, which is SGD's 0.50 on the same window.
  * At `b = 1`, `alpha = 3` it drops from 0.82 to 0.35, well below SGD's 0.54. PSGD drops there too (1.01 -> 0.63),
    with rare-class losses rising for all three optimizers, so this is a late slowdown and not the window edge.
  * At `b = 16` it stays at 0.84-0.82 for `alpha = 2`, where every frontier class is inside the window, and breaks
    from 1.03 to 0.67 at `alpha = 3`.
  * For `alpha <= 1.5` at `b = 1` there is no clear break: Adam stays at 0.53-0.60 while SGD rises by 0.07-0.09 with
    the finite-`C` tail. Inside the `b = 1` window Adam's `t_c` slope is already 0.71-0.74 (PSGD 0.58), because that
    window lies in the crossover rather than on the slope-1/2 branch.
* **Break time.** At `alpha = 2`, `b = 1` the frontier leaves the window between `t = 2.9` (`c* = 6`) and 3.9
  (`c* = 7`), and the local exponent rolls off over more than a decade. At `b = 16` it leaves later: `t = 5.3-5.8` at
  `alpha = 2` and 3.4-4.5 at `alpha = 3`, against 1.6-3.2 at `b = 1`. The plan's
  `1/(eta max(sqrt(1-beta2), b sqrt(eps))) = 33` has no `b` dependence at `eps = 1e-8` and omits `r^2 = 25`. The
  measured break is about `3/(eta r^2 sqrt(1-beta2))`. `beta2` and `eps` were not varied in E2.
* **The `b = 1` frontier at `t = 1000`.** Adam learns 47, 29 and 16 of 64 classes for `alpha` = 1.5, 2 and 3. PSGD
  learns 64, 64 and 21, and SGD 64, 64 and 24.
  * Only `alpha = 2` is shown to be the E4 fixed point.
  * At `alpha = 1.5` the state is nearly stationary (gaps change 2-3% per half decade), and no class has risen above its
    initial loss.
  * At `alpha = 3` the state is far from stationary, and rare-class losses rise for all three optimizers (Adam 45
    classes, SGD 31, PSGD 37 above their initial loss).
  * The prior of the last learned class is 1.3e-3, 7.3e-4 and 2.0e-4.
  * At `alpha = 1.25` Adam learns all 64 classes (`pi_64 = 1.7e-3`), yet its `R(1000) = 0.102` is 3.9x SGD's.
* **Final risk** at `t = 1000`.
  * `b = 16`: Adam is within 1-6% of SGD at `alpha <= 2` (`alpha = 2`: 0.0116 vs 0.0115) and lower at `alpha = 3`
    (0.0047 vs 0.0056).
  * `b = 1` (`alpha` = 1.25 / 1.5 / 2 / 3): Adam 0.102 / 0.131 / 0.074 / 0.012, PSGD 0.072 / 0.060 / 0.038 / 0.017,
    SGD 0.026 / 0.021 / 0.014 / 0.0067.
  * PSGD's risk turns up late in every run. Its minimum is at `t = 130-320`, and `R(1000)` is 4-11% above it at `b = 1`
    and 16-44% at `b = 16`. PSGD runs at Adam's untuned `eta = 0.3`, which E4 shows is too large for it.

**Verdict (H2, H3).** The direction of H2 holds. Inside the window Adam's frontier moves with `s ~ 0.6` against SGD's
1.05, so its risk exponent is 1.66-1.70x SGD's at `C = 64` (PSGD 1.75-1.77x), not 2x. Finite `C` shifts both
exponents but not their ratio. H3's break shows near the window edge where it can be isolated (`alpha = 2`, `b = 1`;
`alpha = 3`, `b = 16`) and is masked by the finite-`C` tail at `alpha <= 1.5`. Its time is about
`3/(eta r^2 sqrt(1-beta2))`. At `b = 1` the late regime is the stall for `alpha` = 1.5-2, and a slower frontier that
is still moving at `alpha = 3`. SGD's exponent is flat over two decades at `alpha >= 2`, but no Adam exponent is flat
over more than ~1.2 decades, so a clean Adam exponent needs `C >> 64`.

### E3 `batch`: batch size at a fixed sample budget (H4)

```
real/bin/python experiments/zipf_scaling.py batch --safety 1     # 41 min (the b = 1 runs are shared with E4)
```

**Setup.** `C = 64`, `alpha = 2`, `beta2 = 0.99`, `eps` in {1e-8, 1e-4}, `b` in {1, 4, 16, 64}. Every run goes to
the same number of samples, `b t = 3000` (in units of `D`). Learning-rate rules:

* Adam and PSGD: fixed `eta = 0.3`, and `eta sqrt(b)`.
* SGD: fixed `eta = 3`, and `eta b`.

Analysis: `experiments/zipf/analysis/e3_batch/e3_analysis.py`, which writes `e3_numbers.json`. Figure:
`figs/experiments/zipf/batch.pdf`. The claims were recomputed independently, and the corrections are included.

**Exact scaling symmetries of the ODE.**

* **SGD with `eta ~ b`** is exactly invariant: `(eta, b, t) -> (eta b, b, t/b)` leaves the drift `eta phi` and the
  diffusion `eta^2 sigma/b` per sample unchanged.
  * The runs learn all 64 classes after 441.2-441.5 samples for `b` = 1-64, and their risk curves agree to 3e-4 for
    `b t >= 300`.
  * Early on, `b` = 16 and 64 deviate: up to 16% at `b t = 0.16`, and up to 4.7% for `b t <= 10` at `b = 64`. This is
    a numerical start-up error. The first record interval, `b t` up to `0.01 b`, is integrated in 1-3 Heun steps,
    because the step controller is limited by `phi` and not by the `chi` forcing, which scales with `eta`. A first
    record time of `0.01/b` would remove it.
* **PSGD with `eta ~ sqrt(b)`** is exactly invariant at `eps = 0`, because its preconditioner
  `(E f^2/b + eps)^-1/2` scales as `sqrt(b)`. The runs at `eps = 1e-8` need 159.7-160.5 samples for all 64 classes,
  and their risks agree to 0.23%. The residual grows with `b`, as `b eps / E f_c^2` reaches 1.3% at `b = 64`.

So in this limit these two rules have no critical batch size. For Adam, every batch effect comes from the `beta2`
memory and from `eps`; for PSGD it comes from `eps`.

**Samples to learn `k` classes** (`b t`; `-` = not within `b t = 3000`). Bold marks the best `b`:

| optimizer, rule | `k` | b = 1 | b = 4 | b = 16 | b = 64 |
|---|---|---|---|---|---|
| Adam `eps` 1e-8, fixed | 8 / 32 / 64 | **5.0** / - / - | 7.6 / 171 / - | 13.6 / **133** / 1039 | 25.7 / 174 / **727** |
| Adam `eps` 1e-8, `sqrt(b)` | 8 / 32 / 64 | 5.0 / - / - | 3.9 / 103 / - | 3.5 / 44 / 496 | **3.4** / **30** / **231** |
| Adam `eps` 1e-4, fixed | 8 / 32 / 64 | **5.2** / 139 / 862 | 8.0 / **119** / **660** | 15.0 / 179 / 919 | 34.6 / 505 / 2585 |
| Adam `eps` 1e-4, `sqrt(b)` | 8 / 32 / 64 | 5.2 / 139 / 862 | 4.1 / 66 / 396 | **3.9** / **52** / **291** | 4.4 / 70 / 379 |
| PSGD `eps` 1e-8, `sqrt(b)` | 64 | 160 | 160 | 160 | 160 |
| SGD, `eta b` | 64 | 441 | 441 | 441 | 441 |
| SGD, fixed | 8 / 64 | **3.7** / **441** | 14 / 1576 | 56 / - | 224 / - |

**H4.**

* **Window optimum.** With `eps = 1e-4` (`b* = sqrt((1-beta2)/eps) = 10`), the samples to learn 24-64 classes are
  smallest at `b = 4` with fixed `eta` and at `b = 16` with `eta sqrt(b)`, which bracket `b*`. Both rules lose at
  `b = 64`, where `pi_c > b eps` holds only for classes 1-9. The risk does not follow under `eta sqrt(b)`: `b = 64`
  ends lowest (`R(3000)` = 0.0101 vs 0.0118 at `b = 16`) and reaches `R <= 0.012` first.
* **`eps = 1e-8`** (`b* = 1000`).
  * Learning all 48-64 classes is cheapest at `b = 64` under both rules. For `k = 64` the cost falls from 1039 to 727
    with fixed `eta`, and from 496 to 231 with `eta sqrt(b)`.
  * But 32-40 classes are cheapest at `b = 16` with fixed `eta`.
  * With `eta sqrt(b)` the final risk is lowest at `b = 16` (0.024 vs 0.031 at `b = 64`). This rule mixes batch size
    with the late-time cost of a large Adam step (`eta` up to 2.4).
* **The best batch grows with the target at fixed `eta`.** For `eps = 1e-8` it is 1 for the first 8-16 classes, 16 for
  32 classes and 64 for 48-64 classes. For `eps = 1e-4` it moves from 1 to 4.
  * This is not a critical batch in the usual sense, because `eta` is not tuned per `b`. The advantage of `b = 1` at the
    head is the expected `sqrt(b)` per-step gain.
  * `b_crit ~ 1/pi_{c*}` is untested with four batch sizes.
* **The stall.** At `b t = 3000` with fixed `eta = 0.3`, Adam (`eps = 1e-8`) learns 29, 43, 64 and 64 classes for
  `b` = 1, 4, 16 and 64 (`R` = 0.074, 0.038, 0.013, 0.0099).
  * The escape is not explained by entering the `beta2` window. `b pi_c > kappa (1-beta2)` holds for classes up to
    6 / 13 / 26 / 52 at `b` = 1 / 4 / 16 / 64, yet at `b = 16` classes 27-64 are learned below the edge.
  * At `b = 4`, 8 classes are still above their initial loss and the gaps are still moving (0.017). Whether a larger
    `b` removes the fixed point or only moves it is open. See the `b = 4` run to `t = 3000` in the diagnostics.
* **SGD at fixed `eta`** wastes samples in proportion to `b`.
* **Not done:** `eta` tuned per `b`, and costs to reach target losses at tuned `eta`. PSGD (`eta = 0.3 sqrt(b)`) learns
  all 64 classes in 160 samples at every `b`, fewer than any Adam run (231).

**Verdict (H4).** Supported for per-class targets, with qualifications.

* With `eps = 1e-4` (`b* = 10`) the samples to learn 24-64 classes are smallest at `b = 4` (fixed `eta`) and
  `b = 16` (`eta sqrt(b)`), and both rules lose at `b = 64`. Under `eta sqrt(b)`, though, `b = 64` reaches the lowest
  risk.
* With `eps = 1e-8` (`b* = 1000`), learning all 48-64 classes is cheapest at `b = 64`.
* At fixed `eta` the best `b` grows as the target moves into the tail.
* `b >= 16` removes the small-batch stall at this sample budget, by a mechanism not identified here.
* The ODE has no critical batch size for SGD with `eta ~ b` or for PSGD with `eta ~ sqrt(b)` (`eps = 0`).

### E5 `bigram`: transfer to a one-hot linear bigram model (direct simulation)

```
real/bin/python experiments/zipf_scaling.py bigram     # 47 min alone (24 configurations, 4 seeds each)
```

**Setup.** This is a plan-style transfer test, and the model is our own simplification of the Kunstner-Bach setting.
Tokens `i ~ pi_i ~ i^-alpha` with `V = 1000` and `alpha` in {1, 1.5}. Each token's next token is a fixed random
permutation `target[i]`, and the logits are `W e_i` with `W = 0` at start (float32). Loss: CE or square
(`|W e_i - e_target|^2/2`). Optimizers:

* Adam with `beta2` in {0.95, 0.999}, `beta1 = 0.9`, `eps = 1e-8`, `lr = 1e-2`;
* SGD with `lr = 1`.

`b` in {1, 16}, `2^22` samples per run. The learning rates are untuned, so absolute times are not comparable across
optimizers. `t_i` is the first step with `ell_i <= ell_i(0)/2`, as a geometric mean over the 4 seeds. Analysis:
`experiments/zipf/analysis/e5_bigram/e5_analysis.py`. Figure: `figs/experiments/zipf/bigram.pdf`, with the predicted
kinks `kappa_beta (1-beta2)/b` as vertical lines.

**Band slopes** of `log t_i` against `log pi_i`, by token rank (the window predicts -1/2, rare events and SGD -1):

| run (CE) | ranks 1-5 | 6-39 | 40-249 | 250-1000 | ranks inside the window |
|---|---|---|---|---|---|
| SGD, `b` 1 / 16, `alpha = 1` | -1.10 / -1.04 | -1.07 / -1.02 | -0.99 / -1.00 | -0.98 / -1.01 | - |
| Adam 0.95, `b = 1`, `alpha = 1` | -0.73 | -0.93 | -1.00 | -1.00 | 1-3 |
| Adam 0.95, `b = 16`, `alpha = 1` | -0.44 | -0.64 | -0.83 | -0.96 | 1-46 |
| Adam 0.999, `b = 1`, `alpha = 1` | -1.04 | -0.89 | -0.87 | -0.93 | 1-85 |
| Adam 0.999, `b = 16`, `alpha = 1` | -0.54 | -0.79 | -0.81 | -0.73 | all |
| Adam 0.95, `b` 1 / 16, `alpha = 1.5` | -0.61 / -0.31 | -0.91 / -0.66 | -1.01 / -0.94 | -1.00 / -0.96 | 1-4 / 1-12 |

The square loss gives the same pattern (Adam 0.95: -0.76, -0.97, -1.01, -1.00 at `b = 1` and -0.27, -0.52, -0.81, -0.96
at `b = 16`, `alpha = 1`).

* **The `beta2` kink transfers.** SGD has slope -1 for every token. Adam with `beta2 = 0.95` has slope -1 below
  `kappa_beta (1-beta2)/b` and is shallower above it (down to about -0.3 to -0.6 at the head). The shallow range grows
  with `b` (from ranks 1-5 at `b = 1` to about 1-40 at `b = 16`), as the kink moves down by `1/b`. In the one-hot model
  each occurrence of a rare token gives one sign-like kick, so its learning time counts occurrences (`~ 1/pi`).
* **The window slope is not cleanly -1/2.** The shallowest bands are -0.3 to -0.6. With `beta2 = 0.999` the head is
  learned during Adam's 1000-step warm-up (no bias correction), and the slopes are -0.7 to -0.9 throughout.
* **No stall.** Every Adam configuration learns all 1000 tokens, except where the `b = 16`, `alpha = 1.5` step budget
  runs out (SGD falls short there too). This fits the E4 mechanism. With one-hot inputs, coordinate `(r, i)` receives
  gradient only when token `i` occurs, so its denominator is never dominated by other classes' background: there is no
  normalization mismatch.
* **A different floor appears with the square loss.** With a constant learning rate, Adam's risk plateaus at about
  1e-4 to 3e-3, where SGD converges to 0, because normalized steps do not shrink near the optimum. With CE the optimum
  is at infinity and no plateau shows within the budget.

**Verdict.** In the bigram model the `beta2`/batch kink in per-token learning times appears qualitatively, as
predicted. The Gaussian model's rare-class stall does not, because it needs dense inputs that couple every class into
every coordinate's normalization.
