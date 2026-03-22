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
- ✓ KEPT **Wider time embedding (time_emb_dim 64→128, MLP hidden fixed at 256)** (0.028565→0.027869, −0.000696): more sinusoidal frequency components gave another solid gain at only ~16k extra params. Returns still positive but slowing.
- ✓ KEPT **Wider time embedding (time_emb_dim 128→256)** (0.027157→0.026774, −0.000383): doubling again at ~32k extra params still helped, though delta is ~half the 64→128 step. Diminishing returns confirmed but width axis now near exhaustion.
  - Follow-on: time_emb_dim 512 — one more doubling to test if returns go negative; low priority
  - Follow-on: 6-layer time MLP — depth vs. width tradeoff; may be more efficient than wider embeddings at this point
- ✓ KEPT **Second ResBlock at bottleneck (mid2: ResBlock(C*2, C*2, D))** (0.026774→0.026249, −0.000526): doubling bottleneck depth at 14×14 resolution at ~26k extra params gave a consistent small gain. Bottleneck capacity is still a productive axis.
- ✓ KEPT **Second encoder ResBlock at 28×28 (enc2: ResBlock(C, C, D))** (0.026249→0.025906, −0.000343): adding depth before the stride-2 downsample gave a small but consistent gain (~26k extra params). Full-resolution encoder depth helps skip connection quality. Returns are diminishing but positive across all depth additions.
- ✓ KEPT **Second decoder ResBlock at 28×28 (dec2: ResBlock(C, C, D))** (0.025906→0.025487, −0.000419): symmetric to enc2, adding depth after skip-connection merge continued the trend — slightly larger gain than enc2 step, suggesting decoder refinement at full resolution is at least as valuable. All four depth additions (mid2, enc2, dec2) have been consistently positive.
- ✓ KEPT **Third bottleneck ResBlock (mid3: ResBlock(C*2, C*2, D))** (0.025487→0.025035, −0.000453): tripling bottleneck depth continued to help at ~26k extra params — delta slightly smaller than mid2 (−0.000526) but still positive. Bottleneck depth has now been probed to three layers with diminishing but consistently positive returns.
- ✓ KEPT **Third encoder ResBlock at 28×28 (enc3: ResBlock(C, C, D))** (0.025035→0.024740, −0.000295): tripling encoder depth before downsampling continued the trend at ~26k extra params — delta smaller than dec2 (−0.000419) but consistently positive.
- ✓ KEPT **Third decoder ResBlock at 28×28 (dec3: ResBlock(C, C, D))** (0.024740→0.024415, −0.000325): tripling decoder depth after skip-connection merge continued the trend at ~26k extra params — delta (−0.000325) slightly larger than enc3 (−0.000295), consistent with dec matching or exceeding enc at each step.
- ✓ KEPT **U-Net skip at 14×14 (enc_14: ResBlock(C*2, C*2, D) + dec_14: ResBlock(C*4, C*2, D))** (0.024415→0.024273, −0.000142): completing a proper U-Net skip at the intermediate resolution gave a small positive gain — delta is the smallest yet (~0.00014), but still consistent. The 14×14 skip mirrors the existing 28×28 skip.
- ✓ KEPT **Second encoder ResBlock at 14×14 (enc_14_2: ResBlock(C*2, C*2, D))** (0.024273→0.023756, −0.000517): adding a second pre-bottleneck layer at 14×14 gave a larger gain than expected (~0.00052 vs ~0.00014 for the skip itself) — notably bigger than the 28×28 enc3/dec3 additions (~0.0003). Deepening from 1→2 layers at this resolution is high-leverage. Running best: 0.023756.
- ✓ KEPT **Second decoder ResBlock at 14×14 (dec_14_2: ResBlock(C*2, C*2, D))** (0.023756→0.023403, −0.000353): symmetric to enc_14_2; continued the pattern of decoder matching encoder gains. Delta (~0.000353) is smaller than enc_14_2 (~0.000517) but still solid — larger than 28×28 enc3/dec3 additions (~0.0003). Running best: 0.023403.
  - Follow-on: third encoder ResBlock at 14×14 (enc_14_3) — the 28×28 level went to 3 layers and all helped; 14×14 is now at 2 encoder / 2 decoder, try 3 encoder
  - Follow-on: third decoder ResBlock at 14×14 (dec_14_3) — symmetric push
  - Follow-on: fourth ResBlocks at 28×28 (enc4/dec4) — low priority; returns clearly diminishing (~0.0003/step)
  - Follow-on: fourth bottleneck ResBlock (`mid4`) — low priority; returns clearly diminishing

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
- ✗ FAILED **Min-SNR-γ weighting (γ=5)** (0.026774→0.026950, +0.000176): scaling each timestep's MSE by `min(SNR(t), 5) / SNR(t)` to focus learning on harder low-SNR steps regressed slightly. The linear β schedule's SNR distribution is apparently already well-matched to this model's capacity — re-weighting doesn't help. Direction exhausted.
- ✗ FAILED **v-prediction** (0.024415→0.174841, +0.150427): severe regression — the linear β schedule on MNIST is well-conditioned enough that ε-prediction works fine. v-prediction's gradient-balancing benefit only matters in harder regimes (larger images, complex schedules). Direction exhausted.

### 5. Training Tricks
- ✓ KEPT **Gradient clipping `clip_by_global_norm(1.0)`** (0.027869→0.027157, −0.000713): suppresses early gradient spikes, consistent small gain. Current best: 0.023403.
  - Follow-on: tighter clip (0.5) — may squeeze out more stability benefit
- Larger batch size (256 or 512) if memory allows

## Strategy Notes

- Make **exactly ONE change** per experiment — isolated, testable hypotheses only
- Changes that improve convergence speed are especially valuable (eval uses few epochs)
- If a direction has been tried and failed twice, move on
- If a direction succeeded, consider pushing it further (e.g., if attention at bottleneck helped, try adding it at an earlier resolution)
- Keep the existing CLI interface intact (don't remove argparse arguments)
- Ensure the code imports remain valid and `sample.py` stays compatible with `ddpm_lib.py`
