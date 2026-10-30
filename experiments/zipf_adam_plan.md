# Handoff: Adam scaling laws for Zipf-imbalanced multiclass GMMs

You are continuing a research project on high-dimensional ODEs for streaming Adam.
Read this whole file before touching code. Work autonomously, but write a progress
log as you go (see "Deliverables") and stop to report at the checkpoints marked **[CHECKPOINT]**.

---------------------------------------------------------------------------------
## 0. Where things are

**Code repo**: `github.com/nmarzz/adam`, branch `main`, HEAD `e470eaf`
("Merge branch 'codex': dense-K/batch mixture ODEs, Haar and multiclass experiments").
Pull first, then create a branch `zipf-scaling`. Never force-push. Never rewrite `main`.

**Paper repo** (Overleaf git, may not be on this machine): `paper/gmm_adam.tex` is the
note on Adam ODEs for Gaussian mixtures. `paper/main.tex` is the main paper; do not edit it.
Relevant labels in `gmm_adam.tex`: `eq:gmm_data`, `eq:O_def`, `eq:gmm_risk`,
`eq:gmm_phi`, `eq:gmm_chi`, `eq:A0_scalar`, `eq:coefficients_from_A`, `eq:gmm_system`,
`eq:dense_ito`, `eq:gmm_iso`, `eq:batch_phi`, `eq:batch_chi`; sections "Mini-batches of fixed
size", "Numerical evaluation of the coefficients", "GPU experiments: Haar rotations and five
classes with mini-batches".

### Two implementations exist. Use the newer one.

| Use this (GPU-ready, `lax.scan`) | Role |
|---|---|
| `discounts.path_layout`, `discounts.draw_paths`, `discounts.stationary_coefficients` | Path Monte Carlo for the coefficients `phi (M+,M)`, `chi (C,M)`, `sigma (M,M)`, `tau (M,M)`; supports batch `b`, class means, `sgd=True`, `stein=True` |
| `dynamics.adam_modes`, `dynamics.sgd_modes`, `dynamics.run_mixture_ode` | Mode ODEs `eq:gmm_system` (Heun), diagonal or dense `K` |
| `simulate.MixtureData`, `simulate.build_mixture_adam`, `simulate.build_mixture_sgd`, `simulate.run_mixture_many` | Direct simulation of Adam/SGD with batch `b` |
| `problems.make_gmm_softmax(priors)` | `C`-class softmax CE, `f(q;c) = softmax(q) - e_c`, risk/error by Monte Carlo over fixed Gaussian draws |
| `experiments/multiclass_batch.py` | **Template to copy**: `ode_and_runs`, `gap`, `frozen_drift`, plotting, `save` |

`gmm.py` and `experiments/gmm_adam.py` are the older CPU-only standalone version (they also have
a beta2→1 "PSGD" baseline, `gmm.psgd_coefficients`, which you may port). Do not extend them.

### Conventions (easy to get wrong)
- Simulators take `lr = eta / D`. ODE time is `t = steps / D`. `record_every = D // record_per_unit`.
- Adam's stability constant is `eps * K_ii` (K-scaled). In the coefficients it enters as
  `eps / (1 - beta2)` added to `S`.
- Batch gradient is `y = (1/b) sum_beta f_beta xi_beta`. In the `beta2 -> 1` limit the drift
  scales like `sqrt(b)` and the diffusion is O(1) in `b` (square-root scaling rule).
- `stationary_coefficients` builds the full Jacobian tensor `(paths, T, b, M, M+)`. For
  `C = M = 128` this is tens of GB. See Phase 0.
- `path_layout(beta1, beta2, window, tol=1e-8)` gives burn ≈ 360 / 1833 / 18412 for
  beta2 = 0.95 / 0.99 / 0.999. For beta2 = 0.999 that is very long; budget accordingly.
- `stein=True` estimates `phi` without the derivative (needs `B` nonsingular). Prefer it when
  `eps` is tiny, because the derivative form has a near-delta term `S^{-1/2} - y^2 S^{-3/2}`.
- Use float64 (`config.enable_x64()`) for ODEs and accumulations. float32 is fine for raw draws.
- PDFs saved with `metadata={"CreationDate": None}`. Use `experiments/plot_style.py`.

---------------------------------------------------------------------------------
## 1. Scientific goal

Kunstner & Bach (NeurIPS 2025) derive scaling laws for GD and sign descent on a linear bigram
model under Zipf's law. They find that at α = 1, GD needs about d^{1-ε} steps and sign descent
about sqrt(d). Stochastic Adam with momentum, mini-batches and cross-entropy is left open. Kunstner
et al. (2024) show empirically that heavy-tailed class imbalance is what makes GD lag Adam.

Our GMM + batch ODEs are exact in the high-dimensional limit for *stochastic* Adam. They keep
finite β1, β2, ε and batch `b`. Use them to derive and test a **per-class timescale law** and the
multiclass scaling laws it implies.

### Model (main setting)
- `C` classes with priors `π_c ∝ c^{-α}`, normalized over `c = 1..C`.
- Means `x̄_c = r · q_c`, with `q_c` orthonormal in a random C-dim subspace and `‖x̄_c‖ = r`.
  Pick `r` so that per-class linear separability is good (Bayes error well below 5% for the
  largest `C`). Report the choice.
- `K = I` (isotropic) for the main runs. Softmax CE with `M = C` logits and no bias.
- Init `Θ0 = 0.1 · N(0, 1/D)`, so `B` is nonsingular for `stein=True`.
- Optional ablation: a shared mean component `u` (`x̄_c = u + r q_c`). In the GMM this acts as a
  learnable bias, so the "unigram" can be learned.

### Hypotheses (back-of-envelope; the experiments decide)
Let `E[f_c^2] ≈ π_c + p_c^2` be the second moment of output `c`'s gradient profile. The
denominator for column `c` is then `S_c ≈ π_c / (b(1-β2)) + eps/(1-β2)`. The class-`c` drift
of `O_cc` per unit ODE time should be:

| Regime | Condition | Rate | Learning time `t_c` |
|---|---|---|---|
| SGD | – | `∝ η π_c` | `∝ 1/π_c` |
| Adam, "many events" | `bπ_c ≫ 1-β2` and `π_c ≫ b·eps` | `∝ η sqrt(b π_c)` | `∝ π_c^{-1/2}` |
| Adam, rare-event (sign-like kicks) | `bπ_c ≪ 1-β2` | `∝ η bπ_c / sqrt(1-β2)` | `∝ 1/π_c` |
| Adam, ε-dominated | `π_c ≲ b·eps` | `∝ η sqrt(b) π_c / sqrt(eps)` | `∝ 1/π_c` |

Consequences to test:

- **H1, per-class law.** `t_c` scales as `π_c^{-1/2}` for Adam inside the window
  `π_c > π_min := max((1-β2)/b, b·eps)`. Outside the window it scales as `π_c^{-1}`. The kinks
  sit at `bπ_c ≈ 1-β2` and `π_c ≈ b·eps`. The β2→1 PSGD baseline should miss the β2 kink.
- **H2, exponent doubling.** `R(t) ≈ Σ_{c: t_c > t} π_c ℓ_c` gives a local exponent
  `2(α-1)/α` for Adam while the frontier class is inside the window. SGD gives `(α-1)/α`.
- **H3, broken power law.** Once the frontier class passes `π_min`, Adam reverts to `(α-1)/α`.
  Up to constants the break is at `t ≈ 1/(η · max(sqrt(1-β2), b·sqrt(eps)))`.
- **H4, batch window.** A larger `b` lowers the β2 cutoff and raises the ε cutoff. The window
  is widest around `b* ≈ sqrt((1-β2)/eps)`. Separately, the critical batch size may grow as the
  frontier moves to rarer classes, like `b_crit ∝ 1/π_{c*(t)}`.
- **H5, rare-class noise floor.** Adam's normalized diffusion is O(1) per column, but the drift
  is only `η sqrt(bπ_c)`. So rare-class logit variance `B_cc` may plateau at about
  `η/sqrt(bπ_c)`, and the learning rate would be limited by the tail.

**Model caveat.** In the Gaussian high-D model the per-coordinate signal is O(D^{-1/2}) against
O(1) noise. There is therefore no deterministic sign-descent regime for frequent classes, unlike
the one-hot bigram. H1–H4 concern the tail. Say this explicitly in the write-up.

---------------------------------------------------------------------------------
## 2. Phase 0: engineering (do first, test everything)

1. **Environment.** `git pull && git checkout -b zipf-scaling`. Confirm the GPU with
   `jax.devices()`. Run the existing `experiments/multiclass_batch.py drift` (small) to confirm
   the stack works.
2. **Scalable coefficients.** Add a new function next to `stationary_coefficients`, or extend it
   with keyword arguments whose defaults keep current behaviour byte-identical:
   - Analytic softmax Jacobian, `∂f_r/∂q_s = p_r(δ_rs - p_s)`. Contract it on the fly; never
     materialize `(paths, T, b, M, M+)`. Or use `stein=True`.
   - Chunk over paths (`lax.map` or a Python loop accumulating sums). Accumulate in float64.
   - **Rare-class variance.** `chi_c`, `phi_cc` and `sigma_cc` are driven by class-`c` events,
     which are rare. Print per-class event counts (`paths × window × b × π_c`) and
     bootstrap/batch-means standard errors for `chi_cc`, `phi_cc` and `sigma_cc`. Require about
     ≥10^3 events or a relative s.e. below ~3% for every class you report on.
   - If that is too expensive, implement a stratified estimator: force the class of one sample
     at well-separated times, spaced more than burn + tail apart, and reweight by `π_c`. Validate
     it against the plain estimator at moderate `π_c`.
   - **Useful alternative: the ψ-route estimator (Appendix A).** It is exact and avoids the
     rare-class problem altogether.
     - Tabulate one scalar function `ψ_r(f)` and its derivative per output, from background
       paths. No rare event has to occur for this.
     - Get `χ` and `φ` from per-class Gaussian integrals over `q ~ N(O_c, B)`.
     - The frequency appears only as the explicit factor `π_c`, so rare classes cost nothing
       extra.
     - `σ̂` is not of this form; keep estimating it from paths. For a rare output that is still
       an event-driven quantity, so check its s.e.
     - A prototype check against the path estimator, at two seeds with 4.8·10^5 samples:
       binary logistic, π = (0.3, 0.7), B = 0.5, O = (−0.4, 0.6), β = (0.9, 0.95), eps = 1e-3.
       `χ` and `φ` agreed to ≤ 0.3% for `b = 1`. A smaller run with `b = 4` agreed to about 1%,
       which is within MC error at that size.
     - Implement both estimators and cross-check them at `C = 5` and `C = 16` before choosing.
       The ψ-route is likely the better default for large `C` and small `π_C`.
   - Unit test: at `C = 5` (the `multiclass_batch.py` mixture) and `b ∈ {1, 16}`, the new
     coefficients must match the old ones within MC error.
3. **Collapsed isotropic ODE.** For `K = I` every mode has `ν = 1` and `w_iso = w_cov = 1`, so
   the mode-summed state `(P, U, O)` is exact. The Itô term becomes `η² σ` because
   `Σ_j w_iso,j / D = 1`. Add a path to `run_mixture_ode` that takes the aggregated initial
   state directly instead of building it from rank-one outer products (`_mode_state`). Check it
   against the mode-resolved ODE at D = 500.
4. **Long horizons.** Scaling laws need log-time.
   - ODE: let `dt` grow (piecewise-constant or geometric). Check against step-halving, and keep
     `dt · η · ‖φ‖ < ~0.2`.
   - Record on a log-spaced grid, in both the ODE and the simulator. Use nested scans with
     segment lengths taken from the grid.
5. **Per-class metrics** from `(B, O)`:
   - `ℓ_c = E[-log p_c(x) | c]` and `err_c`.
   - Chunk over risk draws; `(C, n, C)` blows up at large `C`.
   - Learning time `t_c` = first time `ℓ_c ≤ ½ ℓ_c(0)`. Also record the time when `err_c ≤ 0.5`.
6. **Baselines.** SGD via `sgd=True` + `sgd_modes` (with its own η). PSGD (β2→1, replace `S` by
   its expectation): port from `gmm.psgd_coefficients` if needed.
7. **Regression check.** Re-run one existing configuration, `multiclass_batch.py batch` with
   `b = 4`, and confirm the reported gap is unchanged.

Print wall-clock timings for one coefficient evaluation and one full ODE run as functions of
`C`, `b` and β2. Use them to choose the feasible `C` range.

**[CHECKPOINT A]** Write the timings, test results and the chosen `C`, `D` and path counts to the
log before running sweeps.

---------------------------------------------------------------------------------
## 3. Experiments

All runs go in a new script `experiments/zipf_scaling.py` with subcommands
`validate | rates | timescales | exponents | batch | noise | bigram | plot`.
Outputs are `.npz` files in `figs/experiments/zipf/`; figures are PDFs in the same folder.

Defaults: β1 = 0.9, β2 ∈ {0.95, 0.99, 0.999}, `eps` ∈ {1e-8, 1e-4, 1e-2} (K-scaled),
`b` ∈ {1, 4, 16, 64}. Tune η for Adam and SGD separately on a small grid. Report the chosen η
and keep it fixed across `C` and α unless the experiment says otherwise.

**Picking parameters.** You do not need huge `C` to see the kinks: choose (β2, b, eps) so that
`π_min` falls inside the frequency range. For example, with α = 2, `C` = 16 and β2 = 0.99,
`b = 1` puts the β2 kink at π ≈ 0.01, in the middle of the range.

### E0 `validate`: does the ODE stay accurate with many rare classes?
- **Grid:** `C` ∈ {8, 32, 128 (if feasible)}, α ∈ {1.2, 1.5, 2}, `b` ∈ {1, 8, 32},
  `D` ∈ {2000, 4000, 8000} with `D ≫ C`, 16–32 runs each.
- **Figures:** risk vs predicted risk (total CE and 0-1 error); per-class `ℓ_c(t)` for a few
  classes spanning the frequencies; sup-gap vs `D` (expect about `1/sqrt(D)` per run and
  `O(b/D)` bias).
- **Report:** how the gap depends on `D·π_C` (the expected number of rare-class events).
- **Pass criterion:** mean gap within ~3 s.e. or below 1% of the risk at the largest `D`.
  If it fails for rare classes, diagnose whether the cause is coefficient MC error or finite-D.

### E1 `rates` + `timescales`: the per-class law (H1). Most important.
- **(a) Frozen rates.** At reference states (`Θ ≈ 0`, and a state taken from an ODE trajectory
  midway through training), compute the per-class drift rate of `O_cc` from the coefficients.
  Plot it against `π_c` on log-log axes, with reference slopes ½ and 1 and vertical lines at the
  predicted kinks `π = (1-β2)/b` and `π = b·eps`.
  - Sweep β2, `b` and `eps` to move the kinks; overlay the PSGD and SGD rates.
  - Check two or three rare classes directly with the simulated frozen-parameter Adam drift
    (`frozen_drift` in `multiclass_batch.py`).
- **(b) Dynamic.** Plot `t_c` vs `π_c` from ODE trajectories, for Adam, PSGD and SGD, with
  spot-check simulations.
  - Collapse plot: `ℓ_c` against `t·sqrt(bπ_c)` (Adam window), `t·bπ_c/sqrt(1-β2)` (rare-event)
    and `t·π_c` (SGD).
- **Questions:**
  - Are the slopes ½ and 1?
  - Are the kinks where predicted, up to an O(1) factor? Measure the factor.
  - Does PSGD miss the β2 kink?

**[CHECKPOINT B]** Summarize E0 and E1 in the log with figures. If H1 fails, stop and diagnose
before E2 (e.g. the softmax background `p_c²` or the off-diagonal `O_ck` dominating).

### E2 `exponents`: scaling exponents (H2, H3)
- **Setup:** ODE only (validated in E0), with one or two simulation spot-checks.
  α ∈ {1.25, 1.5, 2, 3}; `C` as large as the timings allow (aim for 256–1024 if the collapsed
  ODE and chunked coefficients make it feasible).
- **Fit:** the local exponent `-d log R / d log t` (CE and error). Compare it with `2(α-1)/α` and
  `(α-1)/α`. Only fit in the window where the frontier `c*(t) ≪ C`, because finite `C` adds a
  late plateau. Show how the window grows with `C`.
- **Break:** check whether the measured break time follows
  `1/(η · max(sqrt(1-β2), b·sqrt(eps)))` as β2, `b` and `eps` vary.

### E3 `batch`: batch-size questions (H4)
- **Measure:** samples (`b × steps`) to reach several target losses deep in the tail, against
  `b`. Do this both at fixed η and at η tuned per `b` (also try square-root scaling η ∝ sqrt(b)).
- **Questions:**
  - Is there a window-optimal `b* ≈ sqrt((1-β2)/eps)` (only visible when `eps` is not tiny)?
  - Does the critical batch size grow as the target loss decreases, and does it track
    `1/π_{c*}`?

### E4 `noise`: rare-class noise floor (H5)
- **Measure:** `B_cc(t)` and its plateau against η and `π_c`; per-class loss plateaus against η.
- **Question:** under Zipf, is the optimal η set by the tail? Compare against SGD.

### E5 `bigram` (optional, outside the Gaussian model)
- **Setup:** direct simulation of Adam (β2 ∈ {0.95, 0.999}, a few `b`) on the Kunstner–Bach
  one-hot linear bigram model, square loss and CE, vocabulary 1k–10k, Zipf α ∈ {1, 1.5}.
- **Question:** do the β2/batch kink in per-class learning times and the broken power law appear
  there too? There is no ODE here; it is a qualitative transfer test.

---------------------------------------------------------------------------------
## 4. Deliverables

1. **Code** on branch `zipf-scaling`:
   - The new estimator and ODE options, all backward compatible.
   - `experiments/zipf_scaling.py`.
   - Tests (`tests/test_zipf_coefficients.py` or a `--selftest` subcommand).
2. **Log:** `experiments/zipf/README.md`. Per experiment: the exact command, wall-clock time,
   key numbers (tables), figure paths, and a 2–4 sentence verdict on the hypothesis.
   Record negative results plainly.
3. **Note:** `paper/zipf_adam.tex` if the paper repo is present; otherwise `code/notes/zipf_adam.tex`.
   - Use `\input{preamble}`, `refs.bib`, `xr-hyper` with `\externaldocument{gmm_adam}`
     (`\ref`, not `\cref`, for external labels).
   - Sections:
     - Model.
     - Per-class rates from `χ` and `S`: derive the regime table from `eq:batch_chi` and the
       stationary `S`, with constants where possible.
     - Experiments E0–E4.
     - Relation to Kunstner & Bach.
     - Limitations: no sign regime, finite `C`, `D ≫ C`.
   - Add bib entries only with verified metadata.
4. **Commits:** small and descriptive, author as configured on the machine, ending with the
   session attribution lines your harness provides. Push the branch to `origin` if credentials
   allow; otherwise say so.

## Appendix A. Explicit-frequency form of the coefficients (the ψ-route)

This is derived from `eq:gmm_phi`, `eq:gmm_chi`, `eq:batch_phi` and `eq:batch_chi`, and it is
exact under the same stationary i.i.d.-lag assumptions. Fix output `r`. At lag `ℓ`, write the
batch average as `y_ℓ = f ξ / b + Y_ℓ`, where `Y_ℓ` is the rest of that batch. Let
`S_r^{(-ℓ)} = S_r - β2^ℓ y_ℓ²` be the denominator without lag `ℓ`. Define

```
ψ_r(f)  = b c_β Σ_ℓ β1^ℓ E[ ξ y_ℓ / sqrt(β2^ℓ y_ℓ² + S_r^{(-ℓ)}) ]        (ξ ~ N(0,1) fresh)
ψ_r'(f) =   c_β Σ_ℓ β1^ℓ E[ ξ² ( S^{-1/2} - β2^ℓ y_ℓ² S^{-3/2} ) ]          (S = β2^ℓ y_ℓ² + S^{(-ℓ)})
```

The expectation is over `ξ`, `Y_ℓ` and `S^{(-ℓ)}`, which have the stationary mixture law. For
`b = 1`, `ψ_r(f) = f w_r(f²)`. `ψ_r` is odd and increasing:

- **Small errors:** linear with slope about `(E f_r² + eps)^{-1/2}`, the preconditioned-SGD
  behaviour.
- **Errors large against the background RMS:** it saturates, the sign-like behaviour.

Applying ψ componentwise,

```
φ = Σ_c π_c Φ_c,      Φ_c = E_{q ~ N(O_{·c}, B)}[ ∇_q ψ(f(q; c)) ]          (M+ × M)
χ = diag(π) Ψ̄,        Ψ̄_{c·} = E_{q ~ N(O_{·c}, B)}[ ψ(f(q; c)) ]           (C × M)
```

So these are SGD's coefficients `E[∇f]` and `diag(π) E[f | c]` with `f` replaced by `ψ∘f`. The
diffusion is `σ̂ = Σ_c π_c E[Ω(f(q;c)) | c]`, but `Ω ≠ ψψ^T`, so estimate it from paths.

**Isotropic `K`, orthogonal means of norm `ρ`** (the main Zipf setting), with
`o_c = O_{·c} ∈ R^C`:

```
dB/dt   = -η Σ_c π_c Sym( B Φ_c + o_c ψ̄_c^T ) + η² σ̂
do_c/dt = -η ( φ^T o_c + ρ² π_c ψ̄_c ),        c = 1..C
```

Each class's logit means feel a shared, prior-averaged restoring operator `φ^T` and their own
forcing `ρ² π_c ψ̄_c`. For softmax the own-logit forcing is about `ρ² π_c |ψ_c(-1)|` before the
class is learned. The hypothesis table in Section 1 is a statement about where `|f| = 1` falls on
the curve `ψ_c`:

- **Linear part:** the `sqrt(bπ_c)` window.
- **Saturated part:** the rare-event regime.
- **Floor `δ_ε`:** the ε regime.

Plot `ψ_c` for a frequent, a middle and a rare class in E1. It is the most direct picture of
the mechanism.

**Implementation sketch.**

1. Simulate background paths once per ODE step, or reuse common random numbers.
2. For each lag `ℓ`, form `Y_ℓ` and `S^{(-ℓ)}` (subtract lag `ℓ`'s own-sample contribution).
3. Evaluate `ψ_r` and `ψ_r'` on a grid of `f` values (`[-1, 1]` for softmax/logistic), with
   fresh `ξ`.
4. Interpolate and integrate per class over `q ~ N(o_c, B)`. Use Gauss–Hermite or fixed
   Gaussian draws; for softmax, `f_r` depends on all `C` logits, so use draws.

The cost is `O(C · grid · paths · window)` for `ψ`, plus the per-class integrals, and the
variance does not grow as `π_c → 0`.

## 5. Ground rules
- Do not modify `main.tex`, `gmm_adam.tex` or existing results/figures. Do not change
  existing function defaults.
- Treat the hypotheses as predictions to falsify, not facts to confirm. If the data disagree,
  report what the data say and propose the corrected law.
- Keep any single run under ~30 min. If something is slower, shrink it and note what was cut.
- Fix seeds and save everything needed to regenerate each figure from its `.npz`.
