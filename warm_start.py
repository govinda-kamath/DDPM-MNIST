#!/usr/bin/env python3
"""
Checkpoint utilities for partial weight transfer across architecture changes.

Two sub-commands:

  save  — snapshot the current best model's weights as a path-indexed .npz
           (run after every promoted checkpoint, using the CURRENT ddpm_lib.py)

    python warm_start.py save --ckpt autorun_best/model_best.eqx \
                               --out  autorun_best/model_best.npz

  load  — create a warm-start .eqx using the NEW ddpm_lib.py, copying weights
           from the .npz where leaf path + shape match, random-init otherwise

    python warm_start.py load --npz  autorun_best/model_best.npz \
                               --out  /tmp/warm_start.eqx

In both cases the model class is imported from ddpm_lib.py in the current
working directory, so the command must be run from the repo root.
"""

import argparse
import sys

import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx

from ddpm_lib import SmallUNet


def _keypath_str(keypath) -> str:
    return ".".join(str(k) for k in keypath)


def cmd_save(args):
    """Load checkpoint into current SmallUNet and dump arrays to .npz."""
    template = SmallUNet(key=jax.random.PRNGKey(0))
    model     = eqx.tree_deserialise_leaves(args.ckpt, template)
    arrays    = eqx.filter(model, eqx.is_array)

    data = {
        _keypath_str(kp): np.array(leaf)
        for kp, leaf in jax.tree_util.tree_leaves_with_path(arrays)
    }
    np.savez(args.out, **data)
    print(f"[warm_start save] {len(data)} tensors → {args.out}")


def cmd_load(args):
    """Build a new SmallUNet and fill matching weights from .npz."""
    old = dict(np.load(args.npz))

    new_model          = SmallUNet(key=jax.random.PRNGKey(0))
    arrays, static     = eqx.partition(new_model, eqx.is_array)

    matched = transferred = total = 0

    def maybe_replace(keypath, leaf):
        nonlocal matched, transferred, total
        total += 1
        key = _keypath_str(keypath)
        if key in old:
            matched += 1
            if old[key].shape == leaf.shape:
                transferred += 1
                return jnp.array(old[key])
            # shape changed (e.g. channels widened) — keep random init
        return leaf

    new_arrays = jax.tree_util.tree_map_with_path(maybe_replace, arrays)
    warm_model = eqx.combine(new_arrays, static)
    eqx.tree_serialise_leaves(args.out, warm_model)

    print(
        f"[warm_start load] {transferred}/{total} tensors transferred "
        f"({matched - transferred} shape-mismatched, "
        f"{total - matched} new) → {args.out}"
    )


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("save", help="snapshot weights as .npz")
    s.add_argument("--ckpt", required=True, help="source .eqx checkpoint")
    s.add_argument("--out",  required=True, help="destination .npz file")

    l = sub.add_parser("load", help="create warm-start .eqx from .npz")
    l.add_argument("--npz", required=True, help="source .npz weight index")
    l.add_argument("--out", required=True, help="destination .eqx checkpoint")

    args = p.parse_args()
    if args.cmd == "save":
        cmd_save(args)
    else:
        cmd_load(args)


if __name__ == "__main__":
    main()
