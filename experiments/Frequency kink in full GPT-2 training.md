# Frequency kink in Adam: full GPT-2-style training

## Goal

Test whether the per-token kink seen with a frozen GPT-2 trunk also appears when the whole model is trained from scratch. The head-only runs (`freqkink.py`, frozen drift) gave the following on raw features:

- The rate is ∝ bπ_c below the kink and flat above it.
- β₂ moves the kink roughly ∝ (1 − β₂), but b moves it only ≈ b^(−1/2).

On centered features they gave:

- The window law √(bπ_c) appears.
- The kink is clean at small b and smears at large b.

The question here is whether the kink survives the following: features that are learned and keep changing, a moving trunk, a learning-rate schedule, contiguous text, and the usual training details (warmup, clipping, β₁ > 0). Keep everything else as close to the head-only setup as possible, so that differences can be attributed.

## What stays the same

- GPT-2 BPE tokenizer and OpenWebText. The π_c distribution, the frequency bins and the never-seen token set are therefore comparable with the head-only runs.
- The held-out stream: the same held-out documents as `data/freqkink` (document order seed 0, positions 35,031–38,481 in shards 0–9). These are excluded from training and evaluated with the same ≥ 512-token context where the context length allows.
- The measurement: per-token held-out loss, own logit and background logit; the own-event gain G_c; frequency bins of width 0.25 dex; rate estimators and fits from `freqkink_analysis.py`.
- Optimizer conventions in the head experiments: β₁ = 0 for the main grid, bias correction on, and ε ∈ {10⁻⁸, 10⁻¹²} logged against √v̂.

## What changes, and how to handle it

### 1. Keeping the G decomposition exact

G works because every output column with no own event receives exactly the same update as a never-seen column. Three things keep this true in full training:

- **Untie the output head from the input embedding.** GPT-2 ties them, but then token c's output row also moves when c is an *input*, which breaks the symmetry. A tied variant can come later as an ablation.
- **Zero-initialize the output head.** All columns then start identical. Zero-init of the readout is standard (μP). At the first step the trunk gets no gradient, and from step 2 it trains normally.
- **Pad the vocabulary from 50,257 to 50,304.** The 47 padding ids never occur as targets, so they are guaranteed background columns independent of the data. They stay inside the softmax. Check at every evaluation that their columns agree exactly, as is done now.

AdamW weight decay, global-norm clipping and any lr schedule act identically on identical columns, so they keep the symmetry. Anything per-token, such as label smoothing or z-loss, must be avoided.

With features h(x) from the current trunk, the gain is

G_c(t) = (w_c − w_u) · Σ_{i: y_i = c} h_i / n_c,

computed elementwise in fp32. The loss splits exactly as ℓ_c = (lse − z_u) − G_c. Note that G now mixes two things:
- the head's own-event forcing, which is the theory's object;
- the trunk making token c's contexts more separable.

The local probes in §5 separate them.

### 2. What b means

b is the number of target tokens per optimizer step, as before. Two consequences:

- **Burstiness.** Contiguous text clusters a rare token's occurrences: a token that appears once in a document often appears several times. A batch of B sequences of length L therefore delivers fewer, larger own events than b iid tokens would.
  - Sample training windows iid from the token stream (random offsets, nanoGPT-style), so different sequences in a batch are independent.
  - For each token, record ρ_c(L): the fraction of length-L windows that contain c.
  - Plot results against both bπ_c (token count) and Bρ_c (sequences per batch containing c). The kink should line up in whichever variable counts independent events.
- **Context length.** Short sequences keep b small enough to put the kink in the measurable range. With L = 256:

| B (sequences) | b (tokens) | π* at β₂=0.99 | π* at β₂=0.9 |
|---|---|---|---|
| 4 | 1k | 1.5e-5 | 1.5e-4 |
| 16 | 4k | 3.8e-6 | 3.8e-5 |
| 64 | 16k | 9.6e-7 | 9.6e-6 |
| 256 | 64k | 2.4e-7 | 2.4e-6 |

(π* = κ(1 − β₂)/b with κ = π/2, i.e. β₁ = 0.) The held-out frequencies resolve down to about 10⁻⁷. Use L = 256 throughout and evaluate held-out positions with the same L, so that the contexts match training.

### 3. Learning-rate schedule and warmup

The kink is a property of the second-moment memory, which is counted in steps (1/(1 − β₂)). η only sets how fast things move. The schedule therefore matters for *measurement*, not for where the kink is:

- **Shape: warmup–stable–decay.**
  - Linear warmup, a long constant phase, and a decay only in the last 10% of steps.
  - All rate measurements are taken in the constant phase, or are normalized by the schedule (next bullet).
  - Cosine decay is not used: the effective step would change continuously during the window being measured.
- **Time axis: cumulative step size.** Use τ(t) = Σ_{s≤t} η_s as the time variable. In the head-only frozen-drift runs the per-step drift was exactly ∝ η, so dG/dτ is the schedule-independent rate. Report results against steps, tokens and τ.
- **Warmup length: fixed in steps,** not tokens, and at least several memories long. Take 1,000 steps, which is 10 memories at β₂ = 0.99. Defining warmup in tokens would give b = 64k only about 15 warmup steps at a 1B-token budget, which is shorter than the v memory.
  - Adam's bias correction already damps the first ~1/(1 − β₂) steps.
  - Warmup on top of that mainly protects the trunk, since attention and LayerNorm are unstable at small b with β₁ = 0.
- **Peak η per b:**
  - Start from the square-root rule η ∝ √b, anchored at a tuned value for b = 16k.
  - Then run a 3-point grid (×0.5, ×1, ×2) per b on 10% of the budget, scored by held-out loss, as in `freqkink.py tune`.
  - Rates are compared after dividing by η or using τ, so η only has to be reasonable, not equal across runs.

### 4. Other training details

- **β₁.** Main grid β₁ = 0, which gives the clean κ = π/2 and matches the head-only runs. Realistic replicate β₁ = 0.9, for which κ_β = 1.43 at β₂ = 0.99 and 0.92 at β₂ = 0.95 (`zipf_adam.tex`).
- **Separate parameter groups for the head and the trunk.** The trunk uses fixed AdamW (β₁ = 0.9, β₂ = 0.95). The head uses the experimental (β₁, β₂, ε). Changing only the head's β₂ moves the predicted kink while the trunk and its features stay the same: this is the cleanest single axis in the design. A full-model β₂ sweep is a second axis.
- **Weight decay** 0 on the head; standard 0.1 on the trunk matrices. Head decay would add a per-column pull toward 0 that competes with the rare-event kicks.
- **Gradient clipping** at global norm 1.0, logging the fraction of clipped steps per run. Clipping is frequent at small b, which effectively lowers η. If more than a few percent of steps are clipped, raise the threshold for that run.
- **Precision:**
  - Trunk in bf16 autocast.
  - Head matmuls, logits, softmax-CE and the head's Adam state in fp32. Rare-token head gradients are tiny and must not be rounded.
  - The CE over 50k classes at b = 64k is about 13 GB of fp32 logits, so compute it in micro-batches with gradient accumulation (one optimizer step per b).
- **Model:** small, so that the small-b, many-step runs are affordable. GPT-2-style with pre-LN, GELU and learned positions; 6 layers, d = 384, 6 heads, L = 256. That is about 11M trunk parameters plus 19M input embedding and 19M head. A 12-layer, d = 768 replicate at one setting checks that nothing depends on model size.
- **Budget:** 1B training tokens per run (about 9 of the 80 cached OWT shards), the same for every b. Step counts range from about 1M (b = 1k) to about 15k (b = 64k).

### 5. Measurements

On a log-spaced step schedule, plus every 5% of training:

- **Held-out pass (4M tokens):**
  - Per-token sums of loss, own logit and padding-id logit.
  - Per-token context sums Σ h_i (C × d, fp32), giving G_c exactly.
  - The common mean of h and its norm relative to the per-token spread. On raw GPT-2 features the shared mean caused the flat plateau, so track how it develops here.
- **Head optimizer state:** per-column RMS and median of √v̂, compared against ε.
- **Local frozen-drift probes:**
  - At about 6 checkpoints (for example 1%, 3%, 10%, 30%, 60% and 100% of training), copy the model.
  - Freeze the trunk, keep the head's (W, v) as they are, and continue for 50M tokens at η_head = 10⁻¹⁰ on fresh training data.
  - The slope dG/dt per bin is the per-token drift *given the current features*: exactly the head-only experiment, run on learned features at that point of training. This separates the optimizer kink (probe) from feature learning (the full-run curves).
- **Burstiness:** ρ_c(L) computed once on the training stream.

### 6. Analysis

- **Rate versus frequency** for each run and probe, with the predicted π* = κ_β(1 − β₂)/b marked. Fit the bend freely (two-segment fit) and report the slopes on both sides.
- **Collapse** in x = bπ_c/(1 − β₂), and in x_ρ = Bρ_c/(1 − β₂). Pool across b and β₂_head, normalizing each run by its own window regime. Use the plateau normalization only if the window side is flat; otherwise normalize by the fitted window-law prefactor.
- **Through training:** use the probe collapse at each checkpoint to see whether the kink location and slopes drift as features are learned. In the full runs, use the time for G to reach θ, against τ.
- **Side-by-side comparison** with the head-only results on the same bins:
  - frozen GPT-2 features, raw;
  - frozen GPT-2 features, centered;
  - learned features at each checkpoint.

## Grid

**Stage 0: infrastructure and checks** (about 1 day of work, a few GPU-hours)

- nanoGPT-style trainer in PyTorch (the `lm/` venv; `torch.compile`).
- Tokenized OWT stored as a uint16 memmap, with held-out documents removed.
- Resumable checkpoints, run as sbatch jobs (the partition limit is 24 h).
- Checks:
  - the padding-id columns agree exactly throughout training;
  - the loss curve at b = 16k is sensible for this model size;
  - the probe slope at 0% training (zero head, random trunk) is linear in t.

**Stage 1: first deliverable**

- b ∈ {1k, 4k, 16k, 64k} × β₂_head ∈ {0.9, 0.99}, with β₁ = 0 on the head, ε = 10⁻⁸ and the trunk fixed. 8 runs.
- Probes at 6 checkpoints each.
- Outputs: the rate-vs-frequency plot and the pooled collapse at each checkpoint.

**Stage 2: realism and controls**

- β₁ = 0.9 on the head (κ_β changes), at 2 values of b.
- Full-model β₂ ∈ {0.95, 0.99}, i.e. trunk and head together, at 2 values of b.
- Head trained with SGD while the trunk uses Adam: no kink is expected (control).
- Tied embeddings: an ablation; the G decomposition is then approximate.
- The 12-layer, d = 768 model at one setting.

## Compute

At about 1.8 × 10⁸ FLOP per token and 1B tokens, a run is about 2 × 10¹⁷ FLOP, which is minutes of H100 time at reasonable utilization. Small b is limited by per-step overhead instead:

- b = 1k is about 1M steps, roughly 3–4 h at about 10 ms per step;
- b = 64k is under 30 minutes.

Stage 1 is therefore about 15–20 GPU-hours including η tuning and probes; a probe is roughly 5 minutes. The node has 3 H100s, so three runs can go in parallel as separate sbatch jobs.

## Risks

- **Small-b instability with β₁ = 0.** Warmup and clipping should handle it. If b = 1k won't train, use L = 128 with B = 8 instead, which keeps b = 1k.
- **Burstiness** may smear the kink when plotted against π_c. The ρ_c axis addresses this.
- **Changing features** make full-run rates hard to interpret. The probes are the main evidence and the full-run curves are supporting.
- **A small model** gives weaker features than GPT-2 small. Rare tokens may never be predictable from context, so G stays dominated by bias-like learning. The 12-layer replicate and the trend in the shared mean of h will show whether this matters.

## Open decisions

1. Is a model of 6 layers, d = 384, L = 256 with 1B tokens per run acceptable, or should it be closer to GPT-2 small (12 layers, d = 768, L = 1024)? The larger model pushes b up to at least 8k per step and makes the small-b runs about 10× more expensive.
2. Is varying β₂ on the head only (trunk fixed) acceptable as the main β₂ axis, with full-model β₂ as a secondary axis?
3. Should the warmup be 1,000 steps for every b, followed by a constant learning rate and decay only in the last 10%?
