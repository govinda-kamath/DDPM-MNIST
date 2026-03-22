"""
DDPM training script for MNIST.

Usage:
    python train.py
    python train.py --epochs 50 --batch-size 128 --lr 2e-4
    python train.py --resume ./checkpoints/model_epoch010.eqx

TensorBoard:
    tensorboard --logdir ./runs
"""

import json
import os
import gzip
import glob
import struct
import argparse
import urllib.request
import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx
import optax
from tensorboardX import SummaryWriter

from ddpm_lib import make_noise_schedule, q_sample, SmallUNet, sample


# CLI args

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs",        type=int,   default=50)
    p.add_argument("--batch-size",    type=int,   default=128)
    p.add_argument("--lr",            type=float, default=2e-4)
    p.add_argument("--seed",          type=int,   default=42)
    p.add_argument("--log-every",     type=int,   default=100,  help="steps between log lines")
    p.add_argument("--ckpt-dir",      type=str,   default="./checkpoints")
    p.add_argument("--keep-ckpts",    type=int,   default=3,    help="number of recent checkpoints to keep (0 = keep all)")
    p.add_argument("--data-dir",      type=str,   default="./mnist_data")
    p.add_argument("--tb-dir",        type=str,   default="./runs")
    p.add_argument("--resume",        type=str,   default=None, help="path to checkpoint .eqx to resume from")
    p.add_argument("--loss-out",      type=str,   default=None, help="write best train/val loss as JSON to this file")
    return p.parse_args()


# data

MNIST_URL = "https://storage.googleapis.com/cvdf-datasets/mnist/"
FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images":  "t10k-images-idx3-ubyte.gz",
    "test_labels":  "t10k-labels-idx1-ubyte.gz",
}

def download_mnist(data_dir):
    os.makedirs(data_dir, exist_ok=True)
    for fname in FILES.values():
        fpath = os.path.join(data_dir, fname)
        if not os.path.exists(fpath):
            print(f"Downloading {fname}...")
            urllib.request.urlretrieve(MNIST_URL + fname, fpath)

def _load_images(gz_path):
    with gzip.open(gz_path, "rb") as f:
        _, n, h, w = struct.unpack(">IIII", f.read(16))
        imgs = np.frombuffer(f.read(n * h * w), dtype=np.uint8).reshape(n, h, w)
    return imgs.astype(np.float32) / 255.0 * 2.0 - 1.0

def load_mnist(data_dir):
    download_mnist(data_dir)
    train = _load_images(os.path.join(data_dir, FILES["train_images"]))
    test  = _load_images(os.path.join(data_dir, FILES["test_images"]))
    return train[:, None, :, :], test[:, None, :, :]  # (60k,1,28,28), (10k,1,28,28)

def make_dataloader(images, batch_size, key):
    n          = len(images)
    images_jnp = jnp.array(images)
    while True:
        key, subkey = jax.random.split(key)
        perm = jax.random.permutation(subkey, n)
        for start in range(0, n - batch_size + 1, batch_size):
            yield images_jnp[perm[start:start + batch_size]]


# checkpointing

def save_checkpoint(model, ckpt_dir, epoch, val_loss, is_best):
    os.makedirs(ckpt_dir, exist_ok=True)
    path = os.path.join(ckpt_dir, f"model_epoch{epoch:03d}.eqx")
    eqx.tree_serialise_leaves(path, model)
    if is_best:
        best_path = os.path.join(ckpt_dir, "model_best.eqx")
        eqx.tree_serialise_leaves(best_path, model)
        print(f"   ** new best  val_loss={val_loss:.8g}  -> {best_path}")
    return path

def prune_old_checkpoints(ckpt_dir, keep):
    if keep <= 0:
        return
    ckpts = sorted(glob.glob(os.path.join(ckpt_dir, "model_epoch*.eqx")))
    for old in ckpts[:-keep]:
        os.remove(old)
        print(f"   removed old checkpoint {old}")

def load_checkpoint(path, template_model):
    return eqx.tree_deserialise_leaves(path, template_model)


# training

def compute_val_loss(model, val_imgs, sched, T, batch_size, key):
    """Compute mean MSE loss over the full validation set (no gradients)."""
    @eqx.filter_jit
    def val_batch(model, x0, t, key):
        x_t, eps = q_sample(x0, t, sched, key)
        eps_pred  = jax.vmap(model)(x_t, t)
        return jnp.mean((eps_pred - eps) ** 2)

    val_jnp    = jnp.array(val_imgs)
    total_loss = 0.0
    n_batches  = 0
    for start in range(0, len(val_imgs) - batch_size + 1, batch_size):
        x0 = val_jnp[start:start + batch_size]
        key, t_key, q_key = jax.random.split(key, 3)
        t = jax.random.randint(t_key, (batch_size,), 0, T)
        total_loss += val_batch(model, x0, t, q_key).item()
        n_batches  += 1
    return total_loss / n_batches


def make_train_step(optimizer, sched, T):
    def loss_fn(model, x0, t, key):
        x_t, eps = q_sample(x0, t, sched, key)
        eps_pred = jax.vmap(model)(x_t, t)
        return jnp.mean((eps_pred - eps) ** 2)

    @eqx.filter_jit
    def train_step(model, opt_state, x0, key):
        key, t_key, q_key = jax.random.split(key, 3)
        t = jax.random.randint(t_key, (x0.shape[0],), 0, T)
        loss, grads = eqx.filter_value_and_grad(loss_fn)(model, x0, t, q_key)
        updates, opt_state = optimizer.update(
            grads, opt_state, eqx.filter(model, eqx.is_array))
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss

    return train_step


def main():
    args = get_args()
    print(f"JAX devices: {jax.devices()}")

    train_imgs, val_imgs = load_mnist(args.data_dir)
    print(f"Train: {train_imgs.shape}  Val: {val_imgs.shape}")

    T     = 1000
    sched = make_noise_schedule(T)

    key       = jax.random.PRNGKey(args.seed)
    key, mkey = jax.random.split(key)
    model     = SmallUNet(key=mkey)

    start_epoch = 0
    if args.resume:
        print(f"Resuming from {args.resume}")
        model = load_checkpoint(args.resume, model)
        base  = os.path.basename(args.resume)
        if base.startswith("model_epoch"):
            start_epoch = int(base[len("model_epoch"):len("model_epoch")+3])
            print(f"Continuing from epoch {start_epoch}")

    n_params = sum(x.size for x in jax.tree.leaves(eqx.filter(model, eqx.is_array)))
    print(f"Parameters: {n_params:,}")

    steps_per_epoch = len(train_imgs) // args.batch_size
    total_steps     = args.epochs * steps_per_epoch
    warmup_steps    = 100
    lr_schedule     = optax.join_schedules(
        schedules=[
            optax.linear_schedule(0.0, args.lr, warmup_steps),
            optax.cosine_decay_schedule(args.lr, total_steps - warmup_steps),
        ],
        boundaries=[warmup_steps],
    )
    optimizer  = optax.adamw(lr_schedule, weight_decay=1e-4)
    opt_state  = optimizer.init(eqx.filter(model, eqx.is_array))
    train_step = make_train_step(optimizer, sched, T)
    key, loader_key = jax.random.split(key)
    loader = make_dataloader(train_imgs, args.batch_size, loader_key)

    writer          = SummaryWriter(log_dir=args.tb_dir)
    best_val_loss   = float("inf")
    best_train_loss = float("inf")
    global_step     = start_epoch * steps_per_epoch

    for epoch in range(start_epoch, args.epochs):
        epoch_loss = 0.0

        for step in range(steps_per_epoch):
            x0_batch = next(loader)
            key, subkey = jax.random.split(key)
            model, opt_state, loss = train_step(model, opt_state, x0_batch, subkey)

            loss_val     = loss.item()
            epoch_loss  += loss_val
            global_step += 1
            writer.add_scalar("loss/step", loss_val, global_step)

            if (step + 1) % args.log_every == 0:
                avg = epoch_loss / (step + 1)
                print(f"epoch {epoch+1:3d}  step {step+1:4d}/{steps_per_epoch}  loss {avg:.4f}")

        avg_loss = epoch_loss / steps_per_epoch
        key, val_key = jax.random.split(key)
        val_loss        = compute_val_loss(model, val_imgs, sched, T, args.batch_size, val_key)
        is_best         = val_loss < best_val_loss
        best_val_loss   = min(best_val_loss, val_loss)
        best_train_loss = min(best_train_loss, avg_loss)

        writer.add_scalar("loss/train_epoch", avg_loss, epoch + 1)
        writer.add_scalar("loss/val",         val_loss, epoch + 1)
        print(f"── epoch {epoch+1:3d} done  avg loss {avg_loss:.8g}  val loss {val_loss:.8g}")

        save_checkpoint(model, args.ckpt_dir, epoch + 1, val_loss, is_best)
        prune_old_checkpoints(args.ckpt_dir, args.keep_ckpts)

        if (epoch + 1) % 5 == 0:
            key, sample_key = jax.random.split(key)
            imgs = sample(model, n_samples=16, key=sample_key, sched=sched, T=T)
            imgs_grid = (np.array(imgs) + 1.0) / 2.0
            writer.add_images("samples", imgs_grid, epoch + 1)

    writer.close()
    print("Training complete.")
    print(f"Best checkpoint: {os.path.join(args.ckpt_dir, 'model_best.eqx')}")
    if args.loss_out:
        with open(args.loss_out, "w") as f:
            json.dump({"best_train_loss": best_train_loss, "best_val_loss": best_val_loss}, f)


if __name__ == "__main__":
    main()
