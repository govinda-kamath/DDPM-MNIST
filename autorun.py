#!/usr/bin/env python3
"""
AutoResearch loop for DDPM-MNIST.

Autonomously proposes and tests improvements to train.py / ddpm_lib.py by:
  1. Evaluating current code (baseline loss)
  2. Calling Claude to make ONE targeted change
  3. Re-evaluating under the same conditions
  4. Keeping the change if it improves loss, reverting otherwise
  5. Logging every experiment to experiments.jsonl

Usage:
    python autorun.py
    python autorun.py --eval-epochs 5 --n-experiments 30
    python autorun.py --eval-epochs 2 --n-experiments 100  # faster, noisier
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

MODIFIABLE_FILES = ["train.py", "ddpm_lib.py"]
BACKUP_SUFFIX    = ".autorun_backup"
BEST_CKPT_DIR    = "./autorun_best"
BEST_CKPT_PATH   = f"{BEST_CKPT_DIR}/model_best.eqx"


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def get_args():
    p = argparse.ArgumentParser(description="AutoResearch loop for DDPM-MNIST")
    p.add_argument("--eval-epochs",   type=int, default=3,
                   help="epochs per eval run (default: 3, ~1-2 min each)")
    p.add_argument("--n-experiments", type=int, default=50,
                   help="number of experiments to run (default: 50)")
    p.add_argument("--timeout",       type=int, default=900,
                   help="per-eval timeout in seconds (default: 900)")
    p.add_argument("--log-file",      type=str, default="experiments.jsonl",
                   help="path to JSONL experiment log (default: experiments.jsonl)")
    p.add_argument("--commit",        action="store_true",
                   help="git-commit improvements automatically")
    p.add_argument("--merge-to",      type=str, default=None, metavar="BRANCH",
                   help="merge autorun branch into this branch at the end (e.g. master). "
                        "Requires --commit. Preserves all per-experiment commits.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

BEST_NPZ_PATH  = f"{BEST_CKPT_DIR}/model_best.npz"
WARM_CKPT_PATH = "/tmp/autorun_warm.eqx"

def promote_checkpoint():
    """Copy the eval checkpoint to the persistent location and snapshot weights as .npz."""
    src = "/tmp/autorun_ckpts/model_best.eqx"
    if not os.path.exists(src):
        return
    os.makedirs(BEST_CKPT_DIR, exist_ok=True)
    shutil.copy2(src, BEST_CKPT_PATH)
    # Snapshot current architecture's weights as a path-indexed .npz so that
    # future experiments with a changed architecture can partially transfer them.
    subprocess.run(
        [sys.executable, "warm_start.py", "save",
         "--ckpt", BEST_CKPT_PATH, "--out", BEST_NPZ_PATH],
        cwd=os.getcwd()
    )

def _make_warm_checkpoint() -> bool:
    """Partially load old weights into the new architecture. Returns True on success."""
    if not os.path.exists(BEST_NPZ_PATH):
        return False
    result = subprocess.run(
        [sys.executable, "warm_start.py", "load",
         "--npz", BEST_NPZ_PATH, "--out", WARM_CKPT_PATH],
        capture_output=True, text=True, cwd=os.getcwd()
    )
    print(f"[eval]  {result.stdout.strip()}")
    return result.returncode == 0 and os.path.exists(WARM_CKPT_PATH)

def _run_training(eval_epochs: int, timeout: int,
                  resume: str | None) -> tuple[float | None, str]:
    cmd = [
        sys.executable, "train.py",
        "--epochs",     str(eval_epochs),
        "--tb-dir",     "/tmp/autorun_tb",
        "--ckpt-dir",   "/tmp/autorun_ckpts",
        "--keep-ckpts", "1",
    ]
    if resume:
        cmd += ["--resume", resume]
    env = os.environ.copy()
    env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout,
            cwd=os.getcwd(), env=env
        )
        output = result.stdout + result.stderr
        losses = re.findall(r"val loss (\S+)", output)
        return (float(losses[-1]) if losses else None), output
    except subprocess.TimeoutExpired:
        return None, "TIMEOUT"
    except Exception as e:
        return None, str(e)

def run_eval(eval_epochs: int, timeout: int) -> tuple[float | None, str]:
    """Run training, resuming from the best checkpoint.

    If the agent changed the model architecture the checkpoint will have
    mismatched tensor shapes. In that case we do a partial weight transfer
    (warm start): copy weights where path + shape match, randomly initialise
    the rest, then resume from the warm-start checkpoint.
    """
    resume = BEST_CKPT_PATH if os.path.exists(BEST_CKPT_PATH) else None
    loss, output = _run_training(eval_epochs, timeout, resume)

    ckpt_incompatible = ("changed shape", "TreePathError", "tree_deserialise", "treedef")
    if loss is None and resume and any(s in output for s in ckpt_incompatible):
        print("[eval]  checkpoint incompatible (structure or shape change) — building warm-start checkpoint")
        if _make_warm_checkpoint():
            loss, output = _run_training(eval_epochs, timeout, WARM_CKPT_PATH)
        else:
            print("[eval]  warm start failed — training from scratch")
            loss, output = _run_training(eval_epochs, timeout, resume=None)

    return loss, output


# ---------------------------------------------------------------------------
# File backup / restore
# ---------------------------------------------------------------------------

def backup_files():
    for fname in MODIFIABLE_FILES:
        if os.path.exists(fname):
            shutil.copy2(fname, fname + BACKUP_SUFFIX)

def restore_files():
    for fname in MODIFIABLE_FILES:
        bak = fname + BACKUP_SUFFIX
        if os.path.exists(bak):
            shutil.copy2(bak, fname)

def drop_backups():
    for fname in MODIFIABLE_FILES:
        bak = fname + BACKUP_SUFFIX
        if os.path.exists(bak):
            os.remove(bak)


# ---------------------------------------------------------------------------
# Claude invocation
# ---------------------------------------------------------------------------

def read_file(path: str) -> str:
    with open(path) as f:
        return f.read()

def read_experiment_log(log_file: str, last_n: int = 6) -> str:
    if not os.path.exists(log_file):
        return "No experiments yet."
    entries = []
    with open(log_file) as f:
        for line in f:
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    entries = entries[-last_n:]
    lines = []
    for e in entries:
        status   = "KEPT    " if e.get("kept") else "REVERTED"
        new_loss = f"{e['new_loss']:.8g}" if e.get("new_loss") is not None else "FAIL"
        lines.append(
            f"  [{status}] baseline={e['baseline_loss']:.8g} → new={new_loss}"
            f"  | {e.get('description', '')[:80]}"
        )
    return "\n".join(lines)

def call_claude(program_md: str, experiment_log: str,
                current_loss: float, eval_epochs: int) -> str:
    """Invoke Claude CLI to propose and implement exactly one improvement."""
    train_src = read_file("train.py")
    ddpm_src  = read_file("ddpm_lib.py")

    prompt = f"""\
You are an autonomous ML researcher running experiments on a DDPM \
(Denoising Diffusion Probabilistic Model) trained on MNIST using JAX + Equinox.

## Research Program
{program_md}

## Current Performance
Baseline validation loss (last epoch of a {eval_epochs}-epoch run, 10k MNIST test set): {current_loss:.8g}
The model is trained on the 60k MNIST training set; validation loss is computed on the held-out 10k test set.

## Experiment History (recent)
{experiment_log}

## Current Code

### train.py
```python
{train_src}
```

### ddpm_lib.py
```python
{ddpm_src}
```

## Your Task
Propose and implement **exactly ONE** specific, targeted improvement to reduce \
the validation loss. Edit train.py and/or ddpm_lib.py directly using your tools.

Rules:
- One change only — small, testable hypothesis
- Must be syntactically valid Python
- Keep the existing CLI interface (all argparse args must still work)
- `sample.py` imports `make_noise_schedule, q_sample, SmallUNet, sample` from \
ddpm_lib — keep those exports compatible
- Begin your final response (after edits) with exactly one line:
  CHANGE: <one-sentence description of what you changed and why>
"""
    result = subprocess.run(
        [
            "claude", "--print",
            "--allowedTools", "Edit,Write,Read,WebSearch,WebFetch",
            "--permission-mode", "bypassPermissions",
            prompt,
        ],
        capture_output=True, text=True, timeout=300,
        cwd=os.getcwd()
    )
    return result.stdout + result.stderr


# ---------------------------------------------------------------------------
# Program reconciliation
# ---------------------------------------------------------------------------

def reconcile_program(description: str, kept: bool,
                      baseline_loss: float, new_loss: float | None,
                      best_loss: float) -> None:
    """Ask Claude to update program.md based on the experiment outcome."""
    program_md  = read_file("program.md")
    outcome     = "KEPT" if kept else "REVERTED"
    loss_str    = f"{new_loss:.8g}" if new_loss is not None else "FAILED (crash/timeout)"
    delta_str   = (f"{baseline_loss - new_loss:+.8g}" if new_loss is not None else "n/a")

    prompt = f"""\
You are maintaining a living research program for DDPM-MNIST experiments.
An experiment just completed. Update program.md to reflect what was learned.

## Experiment result
- Change attempted : {description}
- Outcome          : {outcome}
- Baseline val loss : {baseline_loss:.8g}
- New val loss      : {loss_str}
- Delta             : {delta_str}
- Running best      : {best_loss:.8g}

## Current program.md
{program_md}

## Instructions
Edit program.md to:
1. Mark the direction that was just tried (use ✓ KEPT or ✗ FAILED next to the bullet).
2. If KEPT: note the achieved loss and any follow-on variants worth exploring.
3. If FAILED: briefly note why it likely failed (if inferable) so future experiments avoid it.
4. Reprioritize remaining untried directions if the result changes their expected value.
5. Add a new direction if the result suggests a promising unexplored variant.
6. Keep the file concise — remove or condense stale/exhausted directions.

Do NOT change the Goal or Current Baseline Architecture sections.
Only edit program.md — no other files.
"""
    subprocess.run(
        [
            "claude", "--print",
            "--allowedTools", "Edit,Read",
            "--permission-mode", "bypassPermissions",
            prompt,
        ],
        capture_output=True, text=True, timeout=120,
        cwd=os.getcwd()
    )


# ---------------------------------------------------------------------------
# Logging & git
# ---------------------------------------------------------------------------

def log_experiment(log_file: str, entry: dict):
    with open(log_file, "a") as f:
        f.write(json.dumps(entry) + "\n")

def git_create_branch(branch: str):
    subprocess.run(["git", "checkout", "-b", branch], check=True)

def git_commit(message: str):
    subprocess.run(["git", "add"] + MODIFIABLE_FILES + ["program.md"], capture_output=True)
    subprocess.run(["git", "commit", "-m", message], capture_output=True)

def git_merge(autorun_branch: str, target_branch: str):
    """Merge autorun branch into target, preserving every per-experiment commit."""
    subprocess.run(["git", "checkout", target_branch], check=True)
    subprocess.run(["git", "merge", "--no-ff", autorun_branch,
                    "-m", f"Merge {autorun_branch} into {target_branch}"], check=True)
    subprocess.run(["git", "checkout", autorun_branch], check=True)
    print(f"[git]   merged {autorun_branch} → {target_branch}")


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main():
    args = get_args()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ERROR: ANTHROPIC_API_KEY is not set in the environment.")
        sys.exit(1)

    # Work on a dedicated branch so master stays clean
    branch = f"autorun/{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    git_create_branch(branch)

    print("=" * 64)
    print("AutoResearch: DDPM-MNIST")
    print(f"  branch={branch}")
    print(f"  eval_epochs={args.eval_epochs}  n_experiments={args.n_experiments}")
    print("=" * 64)

    # Initial baseline
    print(f"\n[init] Computing baseline ({args.eval_epochs} epochs) ...")
    baseline_loss, baseline_out = run_eval(args.eval_epochs, args.timeout)
    if baseline_loss is None:
        print("ERROR: baseline training failed. Output:")
        print(baseline_out[-3000:])
        sys.exit(1)
    promote_checkpoint()
    print(f"[init] Baseline loss: {baseline_loss:.8g}  (checkpoint saved to {BEST_CKPT_PATH})")

    best_loss = baseline_loss

    for i in range(args.n_experiments):
        bar = "=" * 64
        print(f"\n{bar}")
        print(f"Experiment {i+1}/{args.n_experiments}   best so far: {best_loss:.8g}")
        print(bar)

        # Save current state
        backup_files()

        # Ask Claude for an improvement (always use latest program.md)
        program_md = read_file("program.md")
        exp_log    = read_experiment_log(args.log_file)
        print("[agent] Calling Claude ...")
        claude_out = call_claude(program_md, exp_log, best_loss, args.eval_epochs)

        # Extract one-line description
        m = re.search(r"CHANGE:\s*(.+)", claude_out)
        description = m.group(1).strip() if m else "(no description)"
        print(f"[agent] {description}")

        # Evaluate new code
        print(f"[eval]  Running {args.eval_epochs} epochs ...")
        new_loss, new_out = run_eval(args.eval_epochs, args.timeout)

        if new_loss is not None:
            delta = best_loss - new_loss
            pct   = delta / best_loss * 100
            print(f"[eval]  new={new_loss:.8g}  baseline={best_loss:.8g}  "
                  f"delta={delta:+.8g} ({pct:+.1f}%)")
        else:
            print(f"[eval]  FAILED  output tail:\n{new_out[-500:]}")

        kept = new_loss is not None and new_loss < best_loss

        if kept:
            print("[keep]  Improvement kept ✓")
            drop_backups()
            promote_checkpoint()   # new weights become the starting point for next experiment
            if args.commit:
                git_commit(f"autorun: {description}")
                if args.merge_to:
                    git_merge(branch, args.merge_to)
            best_loss = new_loss
        else:
            print("[drop]  No improvement — reverting")
            restore_files()
            drop_backups()

        # Log
        log_experiment(args.log_file, {
            "timestamp":     datetime.now().isoformat(),
            "experiment":    i + 1,
            "description":   description,
            "baseline_loss": best_loss if kept else baseline_loss,
            "new_loss":      new_loss,
            "kept":          kept,
        })

        # Update program.md with what was learned
        print("[prog]  Reconciling program.md ...")
        reconcile_program(description, kept, baseline_loss, new_loss, best_loss)

        # Re-read updated program.md for next iteration
        program_md = read_file("program.md")

    # Summary
    improvement = baseline_loss - best_loss
    print(f"\n{'='*64}")
    print("AutoResearch complete!")
    print(f"  Branch           : {branch}")
    print(f"  Initial baseline : {baseline_loss:.8g}")
    print(f"  Best achieved    : {best_loss:.8g}")
    print(f"  Improvement      : {improvement:.8g}  ({improvement/baseline_loss*100:.1f}%)")
    print(f"  Log              : {args.log_file}")

    if args.merge_to:
        if not args.commit:
            print(f"  [warn] --merge-to requires --commit; skipping merge")
        else:
            subprocess.run(["git", "checkout", args.merge_to], check=True)
            print(f"  Merge-to branch  : {args.merge_to} (kept changes merged incrementally)")

    print(f"\n  Plot results with:")
    print(f"    python plot_experiments.py --log {args.log_file}")
    print("=" * 64)


if __name__ == "__main__":
    main()
