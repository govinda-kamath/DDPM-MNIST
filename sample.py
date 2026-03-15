"""
Sample from a trained DDPM checkpoint.

Usage:
    python sample.py
    python sample.py --ckpt ./checkpoints/model_best.eqx --n-samples 25 --out samples.png
"""

import os
import argparse
import numpy as np
import jax
import equinox as eqx
import matplotlib.pyplot as plt

from ddpm_lib import make_noise_schedule, SmallUNet, sample


# CLI

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt",      type=str, default="./checkpoints/model_best.eqx")
    p.add_argument("--n-samples", type=int, default=25)
    p.add_argument("--out",       type=str, default="samples.png")
    p.add_argument("--seed",      type=int, default=0)
    return p.parse_args()


# main

def main():
    args = get_args()

    if not os.path.exists(args.ckpt):
        raise FileNotFoundError(f"Checkpoint not found: {args.ckpt}")

    print(f"Loading {args.ckpt} ...")
    template = SmallUNet(key=jax.random.PRNGKey(0))
    model    = eqx.tree_deserialise_leaves(args.ckpt, template)

    T     = 1000
    sched = make_noise_schedule(T)

    print(f"Sampling {args.n_samples} images ...")
    key     = jax.random.PRNGKey(args.seed)
    samples = sample(model, args.n_samples, key, sched, T)

    imgs = (np.array(samples[:, 0]) + 1.0) / 2.0   # (N, 28, 28) in [0, 1]

    cols = int(np.ceil(np.sqrt(args.n_samples)))
    rows = int(np.ceil(args.n_samples / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.5, rows * 1.5))
    for i, ax in enumerate(axes.flat):
        if i < args.n_samples:
            ax.imshow(imgs[i], cmap='gray', vmin=0, vmax=1)
        ax.axis('off')

    plt.suptitle(f"DDPM samples  (ckpt: {os.path.basename(args.ckpt)})", fontsize=10)
    plt.tight_layout()
    plt.savefig(args.out, dpi=150, bbox_inches='tight')
    print(f"Saved {args.out}")
    plt.show()


if __name__ == "__main__":
    main()
