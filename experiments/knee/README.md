# Series J: why Adam's per-class curves go plateau → cliff → knee → tail — log

Plan: `/home/math/noah.marshall2@MAIL.MCGILL.CA/.claude/plans/looking-at-the-experiments-rustling-shore.md`. Branch `zipf-scaling`.
Hardware: one RTX PRO 6000 Blackwell (shared), JAX (`real/` venv). Run from the repository root:
- simulations: `real/bin/python experiments/knee.py {selftest, probe, run, status}`;
- theory and cached checks: `experiments/knee/analysis/*.py`.

Outputs go to `figs/experiments/knee/` (untracked).

## Question

In the series I `B_classes_*` figures (`figs/experiments/imbalance/analysis/`), Adam's per-class loss goes through four phases:
1. a plateau;
2. a cliff, much steeper than SGD;
3. a knee;
4. a slow tail that looks like SGD.

At `C = 128, alpha = 1, beta2 <= 0.95` the curve is instead a flat stall.

The shape depends on the noise level `kappa`:
- at `kappa = 1` the bend sits near `ell/ell0 ≈ 1e-2`;
- at `kappa = 0.1` the cliff runs 4–5 decades and then turns into a `1/t` tail.

Series I also stopped `C = 128` at `T = 316`, before it had converged.

## Setup

| | |
|---|---|
| classes | `C = 128`, `pi_c ∝ c^{-alpha}`, `alpha ∈ {0, 1, 2}`; plus `C = 32` at `alpha = 0` |
| data | `x = r q_c + sqrt(kappa) xi`, `r = 5`, `kappa ∈ {2, 1, 0.5, 0.25, 0.1}` |
| model | softmax CE, no bias, batch 1 |
| optimizers | Adam: `eta = 0.3`, `beta1 = 0.9`, `beta2 ∈ {0.9, 0.95, 0.99, 0.999}`, `eps ∈ {1e-8, 1e-16}` (K-scaled), no bias correction. SGD: `eta = 3` |
| simulations | `D = max(1000, 20 r'^2)` = 1000 / 1000 / 1000 / 2000 / 5000; 32 runs |
| horizon | from theory, then extended until every class passes a convergence certificate |

We work in primed units: `K = Id`, `r'^2 = 25/kappa`, `eta'_A = sqrt(kappa) eta`, `eta'_S = kappa eta` (the series I exact
rescaling). Further notation:
- `m_c = O_cc - mean_{k != c} O_kc` is the margin.
- `s_c^2` is the logit-difference variance from `B`.
- `ell_c ≈ L_C(m_c, s_c^2)` is the exact equicorrelated-Gaussian loss (Frullani form, `analysis/diagnostics.py`).

## Hypothesis

### As first written (from gmm_adam eq. gmm_iso and zipf_adam's psi route and E4)

- **Shape.** `ell_c = L_C(m_c, s_c^2)`. Adam and SGD trace nearly the same path in `(m, s^2)`, with `s^2 ≈ gamma m^2/r'^2`
  and `gamma ≈ 2`. They differ only in the clock `m_c(t)`.
- **Plateau.** The plateau lasts until `m_c = O(1)`. Its length is the E1 kink law for `t_c`.
- **Cliff (scale-free normalised forcing).** Adam's column normaliser `R_c = sqrt(own + bg + eps)` shrinks together
  with the gradient, so `dm_c/dt` stays roughly constant. Then `ln ell` is linear in `t`.
  - SGD's forcing is `∝ ell`, so for SGD `m ~ ln t` and `ell ~ 1/t`.
- **Knee: three candidate mechanisms.**
  - **Kε (eps floor).** `own + bg < eps`, so `R_c -> sqrt(eps)` and Adam becomes SGD with step `eta'/sqrt(eps)`.
    The tail is `ell pi t = const`, with `R_A/R_S -> G sqrt(kappa)(eta_S/eta_A) sqrt(eps) = G · 3.16e-4`.
  - **Kg (logit-noise geometry).** The loss approaches the population optimum.
  - **K2 (E4 fixed point).** With short memory: `phi_cc mbar_c = r'^2|chi_cc| + lambda_c`, and the gaps are stationary.

### Stage 1 verdict on the cached series I ODE files (revised hypothesis)

`analysis/recheck_cached.py` passes 18 of 28 checks; the log is `figs/experiments/knee/analysis/logs/recheck_all.log`.
Three independent skeptics reproduced the numbers and refuted none of the measured values. They did flag the
selection biases noted below.

| claim | verdict | what the data say |
|---|---|---|
| cached `B, O` are primed order parameters, `B_0 = 0.01 kappa Id` | ✔ | reproduced from the raw files and the code |
| `L_C(m, s^2)` (Frullani, equicorrelated) is exact | ✔ | matches MC to 4 digits |
| `alpha = 0` optimum is Bayes, `W* = r' I` | ✔ | `m* = r'^2`. `ell*(C = 128)/log C` = 1.4e-1 / 9.4e-3 / 2.1e-5 / 6.2e-11 / 2.1e-27 for `kappa` = 2 / 1 / 0.5 / 0.25 / 0.1 |
| cliff: `ln ell` closer to linear in `t` than in `ln t` | partly | true for 75–100% of classes. But the window ends at the argmax of the local exponent, which biases the test toward linear-in-`t`, and at `kappa = 1` the window spans only about 2× in `t`. Median `d ln e/d ln t` is 0.6–0.7 at `kappa = 1` and 1.1–1.5 at `kappa = 0.1` |
| cliff: `dm_c/dt` constant (within 25%) up to the knee | ✘ | it falls 30–60% by the knee: within 25% for 0% of classes at `kappa = 1` and 9–52% at `kappa = 0.1`. The forcing `chi_cc/pi_c` stays flat; the brake is the restoring term `phi^T O`. P1 overpredicts the slope by 2–3.5× at `kappa = 1`, and about half of that comes from using asymptotic `ln L` instead of exact `L_C` |
| margin rate `kappa 0.1 / 1 = sqrt(10)` | partly | exact at `Theta_0`. At matched margins `m = 1→3` it is 3.2–3.6, for window-regime classes only. It grows to 3.8–5.2 for `m = 3→6`, because the restoring term scales as `eta'` while the forcing scales as `eta' r'^2` |
| SGD: `m ~ ln t`, `ell ~ 1/t` | ✔ | `R^2(m ~ ln t)` 0.97–0.99 |
| Kε: `R_c -> sqrt(eps)` by `T` | ✔ | every learned class, `alpha = 1`; `chi_cc/pi_c = -E[g\|c]/sqrt(eps)` (median 0.96–1.00) |
| Kε: reached by `t ≈ 8` | ✘ | only at `C = 32, alpha = 1, beta2 >= 0.99`. Elsewhere at `t ≈ 24–100`; at `C = 128, alpha = 2, beta2 <= 0.99` not by `T = 316` |
| Kε tail: `ell_c pi_c t` constant across classes | ✔ | max/min 1.09–1.19 (near the 2^14-draw MC noise); level 1.3–1.8× naive |
| tail ratio `G` | ✔ where in tail | `G = 1.11–1.19` for every run in a `1/t` tail, independent of `beta2`. `C = 128, alpha = 2` is not in its tail at `T` |
| K2 balance at `kappa = 1, C = 128, alpha = 1, beta2 = 0.95` | ✔ | the end state is a fixed point of the frozen drift. The quoted `|ratio - 1| = 1.7e-6` restates that the margins are stationary: the ratio is algebraically `1 - rate/(forcing + lambda)`. `lambda` ≈ 16% of the forcing. `d ell/dt = 0` is a cancellation of the B-drift (−0.018) against the Itô term (+0.018) |
| gaps stationary on `[T/3, T]` | ✔ | 1.3e-3 per half decade, still falling; just above the planned 1e-3 FIXED floor |
| shape collapse `ell = L_C(m, s^2)` | partly | holds to about 1% at `kappa = 0.1`. At `kappa = 1`, after the plateau, residuals are +0.05 to +0.14 nats (`ell/ell0` 1e-2..0.5) and −0.11 to −0.18 nats (1e-4..1e-2, `C = 128`) for both Adam and SGD. `gamma ≈ 2` holds late at `kappa = 0.1`, not at `t_c` (1.7–4.8) and not in the `C = 128` stalls (1.3, 1.8) |

**Revised hypothesis going into Stage 3:**

1. **The cliff is forcing-driven but braked.**
   - The normalised forcing `chi_cc/pi_c` really is scale-free through the cliff, which is why Adam is so much
     steeper than SGD.
   - But `dm/dt = eta'(r'^2 |chi_cc| - (phi^T O)_cc + ...)` loses 30–60% to the restoring term by the bend.
   - Relative to the forcing, the restoring term is `1/r'^2`, i.e. `∝ kappa`. So the cliff is closest to linear in
     `t` at low noise.
   - P1 is restated with the measured `phi^T O` brake and exact `L_C`, not as "constant rate".
2. **At `kappa ≳ 0.5` the visible bend is the approach to the population optimum (Kg).**
   - At `kappa = 1` it sits at the optimum `ell*/ell0 ≈ 1e-2`. The point of maximum local exponent comes earlier
     (`ell/ell0 ≈ 0.1–0.25`, `m ≈ 5`).
   - P3 compares `m_knee` with `m* = r'^2`, not with `r'^2/2`.
3. **At `kappa ≲ 0.25` the bend is the eps floor (Kε).**
   - There the optimum (≤ 1e-10) is out of reach, and after the bend Adam is exactly SGD with step `eta'/sqrt(eps)`
     (tail ratio `G ≈ 1.15`).
   - The floor arrives later than claimed (`t ≈ 24–100` at `C = 128`), and the `eps = 1e-16` arm moves it.
4. **K2 is the stall mechanism with short memory**, and it predicts new stalls at `alpha = 0`.
   - The frozen drift on the synthetic family `O = m(I - 11^T/C)`, `gamma = 2` is in `analysis/phi_alpha0.py` and
     `figs/experiments/knee/analysis/phi_alpha0.json`. Its zeros are:

     | C | beta2 | 1/C ÷ kink | drift zero `m` | `ell/ell0` there | reading |
     |---|---|---|---|---|---|
     | 128 | 0.9 | 0.18 | 1.75–1.96 | 0.65 | full stall, no class learned |
     | 128 | 0.95 | 0.17 | 2.05–2.35 | 0.6 | full stall |
     | 128 | 0.99 | 0.54 | 7.9–8.2 | 0.07 | stall well above the optimum |
     | 128 | 0.999 | 4.5 | 15.5–15.9 | 0.013 | near-optimal balance, ≈ 1.4 `ell*` |
     | 32 | 0.9 | 0.60 | 15.7–17.3 | 5.8e-3 | near-optimal, ≈ 1.4 `ell*` |
     | 32 | 0.95 | 0.62 | 18.3–19.4 | 5.0e-3 | near-optimal, ≈ 1.2 `ell*` |
     | 32 | 0.99 / 0.999 | 1.8 / 9.9 | none up to `m = 20` | | frequent side: no fixed point |

     These are frozen evaluations on a one-parameter family, not trajectories, so Stage 3a is a direct test.
   - In the rare regime the formula `mbar ≈ r'^2 psi_sat F/(C phi_cc)` gives 1.55 at `C = 128, beta2 = 0.9`, against
     a measured zero at 1.75.
5. **Shape.** Use the exact `L_C`, never the asymptotic `ln(C-1) - m + s^2/2`, which is 1–5 nats off at `kappa = 1`.
   The Adam/SGD collapse at `kappa = 1` carries a ±0.15 nat residual that both optimizers share. It is an
   approximation of the shape function itself, not a difference between the optimizers.

### Predictions for Stage 3 (numbers as above)

- **P1.** Cliff slope = `(d ln L_C/dm) · dm/dt`, with `dm/dt` taken from the frozen drift at the current state.
  Within 25% of the measured slope, up to the bend.
- **P2.** At `kappa ≤ 0.25`:
  - `ell_knee ∝ sqrt(eps)`, so with `eps = 1e-16` the bend drops by about 100× and the cliff runs longer;
  - tail `ell pi t = const`;
  - `G ≈ 1.15`.
- **P3.** At `kappa ≥ 0.5`:
  - the late `ell_c` approaches `ell_c(W*)` from `analysis/optimum.py`, plus an excess `∝ eta'`;
  - `eps` changes nothing.
- **P4.** `alpha = 0, C = 128, beta2 ≤ 0.95` stalls at `ell/ell0 ≈ 0.6`; `beta2 = 0.99` stalls at about 0.07.
  - Counter-test: at `kappa = 0.1, alpha = 2, beta2 = 0.95`, `eps = 1e-16` brings back a stall where `eps = 1e-8`
    has none.
- **P5.** `alpha = 0, C = 32`:
  - `beta2 ≤ 0.95` converges near the optimum without stalling;
  - `beta2 = 0.999` has a window-rate cliff (independent of `beta2`) and then SGD on a re-timed clock.
- **P6.** The late rise at `kappa ≈ 1` is the Itô term. At a fixed point it exactly balances the B-drift.

## Stage 1: theory and cached checks (2026-11-06)

Files in `analysis/`:
- `diagnostics.py`: margins, geometry, the `ShapeTable` for `L_C`, the normaliser decomposition and the Itô split.
- `optimum.py`: the population CE optimum. `alpha = 0` is exact. For `alpha > 0`, `W*` is diagonal plus a row
  constant (an effective bias), and it is MC at `kappa ≥ 1` or the pair bound at `kappa ≤ 0.5`.
- `frozen.py`: psi-route coefficients that mirror `imbalance.adam_plan`, but write their own JAX cache.
- `phi_alpha0.py`.
- `recheck_cached.py`.

Outputs are in `figs/experiments/knee/analysis/{optimum,recheck_cached,phi_alpha0}.json`.

The full-`W` versus ansatz check at `C = 32` passes on the pair objective. Its MC fit overfits the common random
numbers, so it adds nothing beyond that.

## Stage 2: engineering (2026-11-06)

### Changes to `simulate.py` (additive)

- Added `mixture_carry` and `run_mixture_segments`. The latter scans over `[lo, hi)` step bounds with a global
  step index.
- `_mixture_grad` now draws its noise in the data dtype. With f64 data this is bit-identical.

### `experiments/knee.py`

- Lanes are `beta2 × runs`. Keys are `fold_in(PRNGKey(2026), r)`, the same for every `beta2` lane (common
  random numbers).
- Records sit on the grid `t_k = 0.01·10^{k/16}`. A chunk is 8 records and ends on a half decade.
- A checkpoint is written after every chunk, along with a pinned config, a numerics fingerprint and the device.
- Per run and class it records:
  - `loss` and `loss_cv` (control variate, below `s^2 < 6`), `err`;
  - `Bdiag`, `Odiag`, the margins;
  - `vbar`, `nu`, `frac_eps`, `Udiag`.
- It also records the seed means of `B`, `O` and `U`.
- Certificate: FIXED, TAIL, MOVING or UNRESOLVED per (`beta2`, class).

### Review fixes (two independent reviewers)

- SGD groups no longer pin `eps`, which collided across `eps` arms.
- A non-default `beta2` lane set goes into the name.
- Extension never runs a chunk that ends past the cap. Before, a run could overshoot `T_cap` by up to √10.
- Certificate slopes use record-index windows of 9 and 17 records, instead of rounded-time masks that dropped
  endpoints.
- The observables use `Precision.HIGHEST`; f32 had been getting TF32.
- The fingerprint now covers `zipf_scaling.py`, `lane_beta2`, `Runner` and `_priors`.
- The device and jaxlib are pinned.
- MC draws are raised from 2^14 to 2^16. At 2^14 the per-class error in late states was 2–6%.

`experiments/imbalance.py selftest` now fails one check: the E2 `C = 64` Adam ODE regression. That check runs only
the ODE path, which does not import `simulate.py`, so this series cannot cause it. The likely cause, not yet
verified, is `2fd4407`, which changed the psi plan for `C > 32`.

### Self-tests (`real/bin/python experiments/knee.py selftest`, 121 s): 30 pass, 0 fail, 5 flags

What passes:
- The default path is bit-identical to pinned `0db8cce:simulate.py` (40 arrays).
- Segments match `run_mixture_at` bitwise, and so does the chunk executable.
- Resume is bitwise for Adam and SGD × f64 and f32 × `U ∈ {1, 4}`, and across processes.
- `U = 4` vs `U = 1`: Adam is bitwise; SGD in f64 differs by 8.5e-16.
- A traced `beta2` lane matches a scalar one to 2.8e-15.
- The diagnostics match a numpy recompute to 2e-15.
- The new default of 2^16 draws matches an independent 2^19 reference:
  - plain loss within 1.2–2.4% (max over 128 classes);
  - `loss_cv` within 1e-6 to 4e-4 at `kappa = 0.1`, and 1.7% at `kappa = 1` late states (above the
    control-variate gate).

The 5 flags record the error of the 2^14-draw estimator used in series I (2.4–5.8% per class).

Two independent reviewers reran the self-tests and an adversarial test script. They found:
- no defect in the step index, the key order, zero-length segments or checkpoint atomicity;
- the six issues fixed above.

### Probe (`figs/experiments/knee/probe.json`, idle GPU, 4000 steps, best of 3)

µs per step for all lanes together:

| opt, lanes | D | f64 U=1 | f64 U=4 | f32 U=1 | f32 U=4 |
|---|---|---|---|---|---|
| Adam C=128, 128 | 1000 | 686 | 690 | 327 | 257 |
| Adam C=128, 128 | 5000 | 3304 | 3379 | 1958 | 1318 |
| SGD C=128, 32 | 1000 | 98 | 134 | 36 | 23 |
| SGD C=128, 32 | 5000 | 345 | 430 | 79 | 50 |
| Adam C=32, 128 | 1000 | 232 | 401 | | |

- `U = 16` is never faster.
- Compile times: f64 Adam at `U = 1` takes 51 s, at `U = 4` 4 s (both cached afterwards).
- Metrics take 15 s per chunk at `C = 128` with 128 lanes (2^16 draws).
- Choice: f64 for everything in S3a and S3b; Adam at `U = 4`, SGD at `U = 1`.
- f32 gives only a 2–2.7× speedup. It is reserved for the S3c `eps = 1e-8` groups, and only after passing the
  f32/f64 gate at `alpha = 1, kappa = 1`.

### Budget (f64, horizon rule; GPU-h at `T0` / at the cap)

| stage | at T0 | at cap |
|---|---|---|
| S3a `alpha = 0` (C = 128 and 32, both eps) | 2.3 | 4.2 |
| S3b `alpha = 1` (both eps) | 7.6 | 24 |
| S3c `alpha = 2` (eps = 1e-8) | 127 | 313 |

S3c is far over the plan's 60 GPU-h gate. Its cost is driven by the rarest classes (`pi_128 = 3.7e-5`, so
`t_S ≈ 2400 kappa^0.4`) and by `D = 5000` at `kappa = 0.1` (46 GPU-h at `T0`). The gate's first step is to drop
`kappa = 0.5` and 2, which leaves about 98 GPU-h in f64, or about 40 in f32. A decision is still needed before S3c
starts.

## Stage 3a: alpha = 0 (started 2026-11-06)

Driver: `experiments/knee/s3a.sh`, log in `figs/experiments/knee/logs/s3a.log`. It runs:
- `C = 128` and `C = 32`, `kappa ∈ {2, 1, 0.5, 0.25, 0.1}`, `eps ∈ {1e-8, 1e-16}`, 32 runs, with SGD
  references;
- a `D = 2500` finite-D check at `C = 128, kappa = 0.1`;
- a `D = 4000` check of the `beta2 = 0.999` lane at `C = 32, kappa = 1`.

**Certificate change (2026-11-06, `919b19d`).** On the first group (`C = 128, kappa = 2, eps = 1e-8`) the
planned FIXED test failed on visibly flat stalls:
- `ell/ell0` stayed at 0.8359–0.8364 over two decades with slope 0.000, yet only 20–45 of 128 classes certified.
- The cause is the gap statistic: it is the maximum over 127 seed-mean gaps, about 5% from run noise. Its threshold
  came from the much less noisy class-averaged margin, so the test was a coin flip.

FIXED now requires both of the following, at `T` and at `T/sqrt(10)`:
- the signed relative half-decade change of the seed-mean margin is within `max(1e-3, 3 SE)` of zero, with `SE`
  its paired bootstrap error over runs;
- `|d ln ell/d ln t| < 0.02`.

Result at `T = 3162` by `beta2` = 0.9 / 0.95 / 0.99 / 0.999: 125 / 120 / 84 / 0 FIXED. I restarted the driver
(bit-identical resume) so that later groups stop once they certify.

**First readings (`C = 128, alpha = 0, kappa = 2, eps = 1e-8`, `T = 3162`):**
- `beta2 = 0.9, 0.95`: a joint stall at `ell/ell0 = 0.836` and 0.805, with median margin 0.9 and 1.1.
- `beta2 = 0.99`: a stall at 0.373 (margin 4.8).
- `beta2 = 0.999`: the loss falls to 0.263 at `T ≈ 316`, then rises to 0.297 at 3162 while the margins keep
  growing (19 against the optimum's `m* = 12.5`). This is the late rise of P6.
- The optimum is at `ell*/ell0 = 0.141`.
