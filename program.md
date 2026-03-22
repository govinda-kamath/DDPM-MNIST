# AutoResearch Program: DDPM-MNIST

## Goal

Minimize the MSE validation loss (noise prediction error) on MNIST.
The metric is the **best validation loss across epochs** in a short fixed-length training run.
Lower is better.

## Current Baseline Architecture

- **Model**: `SmallUNet` — sinusoidal time embeddings, GroupNorm, ResBlocks, stride-2 downsampling, nearest-neighbor upsample, skip connections
  - `base_channels=16`, `time_emb_dim=64`, ~80k parameters
- **Noise schedule**: Linear β schedule, β₁=1e-4 → β_T=0.02, T=1000 steps
- **Loss**: Simple MSE on predicted noise ε
- **Optimizer**: Adam, cosine decay lr=2e-4→0 over all steps (loss: 0.0311)
- **Batch size**: 128

## Research Directions

Explore these roughly in order of expected impact:

### 1. Noise Schedule
- ✗ FAILED **Cosine schedule** (tried twice): initially helped (0.1166→0.0791) but at the current stronger baseline (0.0340) it regressed to 0.0528 (+0.0188). Linear β schedule is better for this model/dataset combination at this loss level — cosine likely over-smooths SNR distribution for 28×28 MNIST. Direction exhausted.

### 2. Model Capacity
- ✗ FAILED **base_channels 16→32** (crash/timeout): ~4x parameter count likely exceeded memory or time budget; skip 64 as well
- ✗ FAILED **Self-attention at bottleneck (14×14, 32ch, single-head)** (crash/timeout): adding `SelfAttention2d` at the bottleneck crashed; likely memory/compute overhead of attention on 14×14 maps (196 tokens) with 32 channels is too large for the budget. Try 7×7 spatial resolution instead if re-attempting.
- ✓ KEPT **Deeper time embedding MLP (3-layer: sinusoidal→256→256→256)** (0.035414→0.031385, −0.004029): adding `t_dense3: Linear(256→256)` gave a solid gain at ~65k extra params. Time conditioning is a high-leverage axis.
- ✓ KEPT **4-layer time MLP (sinusoidal→256→256→256→256)** (0.031385→0.029530, −0.001855): another ~65k params, consistent improvement — deeper time MLPs keep helping.
- ✓ KEPT **5-layer time MLP (sinusoidal→256→256→256→256→256)** (0.029530→0.028565, −0.000965): gains are diminishing (~1/2 the delta of the 4-layer step) but still positive.
- ✓ KEPT **Wider time embedding (time_emb_dim 64→128, MLP hidden fixed at 256)** (0.028565→0.027869, −0.000696): more sinusoidal frequency components gave another solid gain at only ~16k extra params. Returns still positive but slowing. Current best: 0.027869.
  - Follow-on: time_emb_dim 128→256 — doubling again costs ~32k more params in t_dense1; diminishing returns likely but worth one more step
  - Follow-on: 6-layer time MLP — depth vs. width tradeoff; lower priority now that width was validated
- Additional ResBlock in the encoder or decoder path

### 3. Optimizer & Learning Rate
- ✓ KEPT **AdamW weight_decay=1e-4** (0.0301→0.0295, −0.0006): modest but consistent gain; regularization helps even in short runs.
  - Follow-on: **AdamW + warmup + cosine decay** combined — all three improvements stacked; highest-priority next experiment
  - Follow-on: stronger weight decay (1e-3) — see if more regularization helps further
- ✓ KEPT **Cosine LR decay** (0.0340→0.0311, −0.0029): `optax.cosine_decay_schedule` from lr=2e-4 to 0 over all steps. Helps converge faster within short training run.
- ✓ KEPT **Warmup + cosine decay** (0.0301→0.0289, −0.0012): 100-step linear warmup 0→2e-4 via `optax.join_schedules`, then cosine decay. Reduces early gradient instability.
  - Follow-on: **AdamW + warmup + cosine** combined — stack weight decay on top of the now-proven warmup schedule
  - Follow-on: longer warmup (200–500 steps) — may help further if 100 steps was still too short
- Slightly higher lr (4e-4) with warmup + cosine — small models can often train faster with higher peak lr

### 4. Loss Formulation
- **SNR-weighted loss**: weight each step by min(SNR, 5) / SNR (Hang et al. 2023). Lower priority now that cosine schedule is ruled out; still worth trying with linear schedule.
- **v-prediction** parameterization instead of ε-prediction: model predicts v = √ᾱ·ε − √(1−ᾱ)·x₀

### 5. Training Tricks
- **Gradient clipping** (e.g. global norm ≤ 1.0) — can stabilize early training
- Larger batch size (256 or 512) if memory allows

## Strategy Notes

- Make **exactly ONE change** per experiment — isolated, testable hypotheses only
- Changes that improve convergence speed are especially valuable (eval uses few epochs)
- If a direction has been tried and failed twice, move on
- If a direction succeeded, consider pushing it further (e.g., if attention at bottleneck helped, try adding it at an earlier resolution)
- Keep the existing CLI interface intact (don't remove argparse arguments)
- Ensure the code imports remain valid and `sample.py` stays compatible with `ddpm_lib.py`
