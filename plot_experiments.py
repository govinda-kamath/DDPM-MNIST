#!/usr/bin/env python3
"""
Plot loss progression across autorun experiments.

Usage:
    python plot_experiments.py
    python plot_experiments.py --log experiments.jsonl --out autorun_results.png
"""

import argparse
import json
import os
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


def load_log(path: str) -> list[dict]:
    entries = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def plot(entries: list[dict], out: str):
    xs_kept     = []
    ys_kept     = []
    xs_reverted = []
    ys_reverted = []
    best_xs     = [0]
    best_ys     = [entries[0]["baseline_loss"]]

    for e in entries:
        n        = e["experiment"]
        new_loss = e.get("new_loss")
        kept     = e.get("kept", False)

        if new_loss is not None:
            if kept:
                xs_kept.append(n)
                ys_kept.append(new_loss)
            else:
                xs_reverted.append(n)
                ys_reverted.append(new_loss)

        best_xs.append(n)
        best_ys.append(min(best_ys[-1], new_loss) if (kept and new_loss) else best_ys[-1])

    fig, ax = plt.subplots(figsize=(12, 5))

    # Best-so-far staircase
    ax.step(best_xs, best_ys, where="post", color="#1f77b4",
            linewidth=2, zorder=2, label="best loss so far")

    # Individual experiment results
    ax.scatter(xs_reverted, ys_reverted, color="#d62728", marker="x",
               s=60, zorder=3, linewidths=1.5, label="reverted")
    ax.scatter(xs_kept, ys_kept, color="#2ca02c", marker="o",
               s=60, zorder=3, label="kept")

    # Annotate kept changes with short descriptions
    for e in entries:
        if e.get("kept") and e.get("new_loss") is not None:
            desc = e.get("description", "")
            short = textwrap.shorten(desc, width=28, placeholder="…")
            ax.annotate(
                short,
                xy=(e["experiment"], e["new_loss"]),
                xytext=(6, 4), textcoords="offset points",
                fontsize=6.5, color="#2ca02c",
                arrowprops=None,
            )

    ax.set_xlabel("Experiment #", fontsize=11)
    ax.set_ylabel("Avg training loss (MSE)", fontsize=11)
    ax.set_title("AutoResearch: DDPM-MNIST loss progression", fontsize=13)
    ax.set_ylim([0.01, 0.05])
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Summary box
    initial = entries[0]["baseline_loss"]
    final   = best_ys[-1]
    pct     = (initial - final) / initial * 100
    n_kept  = sum(1 for e in entries if e.get("kept"))
    summary = (f"Experiments: {len(entries)}  |  Kept: {n_kept}  |  "
               f"Loss: {initial:.4f} → {final:.4f}  ({pct:.1f}% ↓)")
    ax.text(0.5, -0.13, summary, transform=ax.transAxes,
            ha="center", fontsize=9, color="#555555")

    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved figure to {out}  ({len(entries)} experiments, {n_kept} kept)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--log", default="experiments.jsonl",
                   help="path to JSONL experiment log (default: experiments.jsonl)")
    p.add_argument("--out", default="autorun_results.png",
                   help="output image path (default: autorun_results.png)")
    args = p.parse_args()

    if not os.path.exists(args.log):
        print(f"Log file not found: {args.log}")
        return

    entries = load_log(args.log)
    if not entries:
        print("Log is empty.")
        return

    plot(entries, args.out)


if __name__ == "__main__":
    main()
