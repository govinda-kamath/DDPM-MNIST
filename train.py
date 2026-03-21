"""
DDPM training script for MNIST.

Usage:
    python train.py
    python train.py --epochs 50 --batch-size 128 --lr 2e-4
    python train.py --resume ./checkpoints/model_epoch010.eqx

TensorBoard:
    tensorboard --logdir ./runs
"""

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
    imgs  = np.concatenate([train, test], axis=0)   # 70k total
    return imgs[:, None, :, :]

def make_dataloader(images, batch_size, key):
    n          = len(images)
    images_jnp = jnp.array(images)
    while True:
        key, subkey = jax.random.split(key)
        perm = jax.random.permutation(subkey, n)
        for start in range(0, n - batch_size + 1, batch_size):
            yield images_jnp[perm[start:start + batch_size]]


# checkpointing

def save_checkpoint(model, ckpt_dir, epoch, loss, is_best):
    os.makedirs(ckpt_dir, exist_ok=True)
    path = os.path.join(ckpt_dir, f"model_epoch{epoch:03d}.eqx")
    eqx.tree_serialise_leaves(path, model)
    if is_best:
        best_path = os.path.join(ckpt_dir, "model_best.eqx")
        eqx.tree_serialise_leaves(best_path, model)
        print(f"   ** new best  loss={loss:.4f}  -> {best_path}")
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

    train_imgs = load_mnist(args.data_dir)
    print(f"Train: {train_imgs.shape}  [{train_imgs.min():.2f}, {train_imgs.max():.2f}]")

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

    optimizer  = optax.adam(args.lr)
    opt_state  = optimizer.init(eqx.filter(model, eqx.is_array))
    train_step = make_train_step(optimizer, sched, T)

    steps_per_epoch = len(train_imgs) // args.batch_size
    key, loader_key = jax.random.split(key)
    loader = make_dataloader(train_imgs, args.batch_size, loader_key)

    writer      = SummaryWriter(log_dir=args.tb_dir)
    best_loss   = float("inf")
    global_step = start_epoch * steps_per_epoch

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

        avg_loss  = epoch_loss / steps_per_epoch
        is_best   = avg_loss < best_loss
        best_loss = min(best_loss, avg_loss)

        writer.add_scalar("loss/epoch", avg_loss, epoch + 1)
        print(f"── epoch {epoch+1:3d} done  avg loss {avg_loss:.4f}")

        save_checkpoint(model, args.ckpt_dir, epoch + 1, avg_loss, is_best)
        prune_old_checkpoints(args.ckpt_dir, args.keep_ckpts)

        if (epoch + 1) % 5 == 0:
            key, sample_key = jax.random.split(key)
            imgs = sample(model, n_samples=16, key=sample_key, sched=sched, T=T)
            imgs_grid = (np.array(imgs) + 1.0) / 2.0
            writer.add_images("samples", imgs_grid, epoch + 1)

    writer.close()
    print("Training complete.")
    print(f"Best checkpoint: {os.path.join(args.ckpt_dir, 'model_best.eqx')}")


if __name__ == "__main__":
    main()
