# Frequency kink in Adam: logistic regression on pretrained embeddings

## Goal

Test whether the per-class rate law from the Gaussian-mixture analysis (Section 9.3) shows up on real language data. In the window regime the initial per-class rate grows linearly with b·π\_c. In the rare-event regime (b·π\_c ≪ 1 − β₂) a class's own samples dominate its second-moment denominator and the kicks become sign-like. The two laws cross at b·π\_c ≈ κ\_β (1 − β₂)/F², with κ\_β = π/2 when β₁ = 0. The thesis is only that a kink exists. On real data its location is not expected to match the prediction, so the kink is located from the data and the predicted value is overlaid for comparison. No ODE comparison.

## Setup

- Features: frozen embeddings from a pretrained LM (final hidden state at each position), e.g. GPT-2 small (D = 768). Precompute and cache them.
- Model: a single linear softmax head trained from zero on next-token prediction with cross-entropy.
- Data: a fixed token stream (e.g. a slice of OpenWebText) so that π\_c spans roughly 10⁻² to 10⁻⁷.
- Optimizer: Adam with β₁ = 0, β₂ ∈ {0.9, 0.95, 0.99} (capped at 0.99 so the memory 1/(1 − β₂) stays short relative to D), bias-correction choice held fixed across runs. ε is swept over {10⁻⁸, 10⁻¹²}: rare-token gradients are tiny, so 10⁻⁸ may already be the denominator floor and put those tokens in the ε regime (rate ∝ π\_c/√ε, edge near π\_c ≈ bε) rather than the rare-event one. Log per-output √v against ε to check.
- Control: an SGD (and optionally preconditioned-SGD) baseline, which should show no β₂ kink.
- Basis check: repeat one configuration on randomly rotated embeddings to separate the basis effect from the frequency effect.

## Moving the kink with batch size

The kink sits at π\* ≈ κ(1 − β₂)/b, where b counts tokens per step. Sweep b ∈ {32, 128, 512, 2048}. At β₂ = 0.99 this moves π\* from about 5 × 10⁻⁴ to 8 × 10⁻⁶, across the bulk of the vocabulary. Crossing b with the three β₂ values gives two independent ways of moving the same threshold.

## Measurements

- Per-token loss ℓ\_c(t) on a held-out stream, logged on a log-spaced schedule from the first steps to the end of training.
- Per-token initial rate: slope of ℓ\_c over an early window, and time to half-loss (ℓ\_c ≤ ℓ\_c(0)/2) as a rate proxy that holds through the whole run.
- Empirical π\_c from the training stream; bucket tokens into log-frequency bins to reduce noise.
- Step size matched across runs by tuning η on total loss at a fixed sample budget, or by a small grid per run.

## Analysis

- Self-normalized ratios: for each token, divide its rate under a configuration by its rate under a reference (largest b·π\_c, i.e. deep in the window). Token-specific factors such as context predictability and embedding geometry cancel.
- Collapse test: plot the ratio against x = b·π\_c/(1 − β₂) for all runs. A kink is supported if the ratios bend and collapse onto one curve in x; fit the bend location (piecewise-linear in log–log) and mark the predicted x ≈ κ beside it. A collapse across both the b sweep and the β₂ sweep is the main evidence.
- Slopes: fit log-rate against log π\_c on each side of the predicted kink and compare to the window law (slope 1) and to the rare-event regime.
- Through training: repeat the collapse at several checkpoints to see whether the effect persists past initialization.

## Risks

- Anisotropic embeddings may make Adam's diagonal preconditioning dominate; the rotated control addresses this.
- Very rare tokens give noisy loss estimates; use frequency bins and enough held-out tokens.
- Large b reduces the number of steps at a fixed sample budget; compare runs on samples seen as well as on steps.
- The background kink at π\_c ≈ 1/C² (C ≈ 50k, so about 4 × 10⁻¹⁰) sits below the range studied and should not interfere.

## First deliverable

One β₂ (0.99), four batch sizes, two ε values (10⁻¹² first, then 10⁻⁸), Adam plus SGD, raw embeddings: a per-token rate versus frequency plot and the collapse plot.
