"""Local MLX training for draft model validation.

Trains a tiny AR draft model on cached teacher outputs.
Runs on Mac (Apple Silicon) — proves the concept before scaling to A100.

Usage:
    python local/train_local.py \
        --cache_dir ./cache_local \
        --num_epochs 3 \
        --batch_size 4
"""

import argparse
import json
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

from draft_mlx import DraftMLXConfig, DraftModelMLX


def load_cached_dataset(cache_dir: str):
    """Load all cached teacher outputs into memory."""
    cache_path = Path(cache_dir)
    manifest = json.loads((cache_path / "manifest.json").read_text())

    samples = []
    for entry in manifest:
        data = np.load(cache_path / entry["file"])
        samples.append({
            "cond_hidden": mx.array(data["cond_hidden"]),  # (L_cond, H)
            "cb0_tokens": mx.array(data["cb0_tokens"].astype(np.int32)),  # (target_len,)
        })

    print(f"Loaded {len(samples)} samples from {cache_dir}")
    return samples


def make_training_batch(samples, batch_indices):
    """Create a padded batch from samples.

    For simplicity, we pad to max length in batch. In production
    you'd bucket by length.
    """
    batch_cond = []
    batch_tokens = []
    batch_targets = []

    for idx in batch_indices:
        s = samples[idx]
        cond = s["cond_hidden"]     # (L_cond, H)
        tokens = s["cb0_tokens"]    # (T,)

        # Input: tokens[:-1], Target: tokens[1:]  (standard AR shift)
        batch_cond.append(cond)
        batch_tokens.append(tokens[:-1])
        batch_targets.append(tokens[1:])

    # Pad to max length in batch
    max_cond = max(c.shape[0] for c in batch_cond)
    max_tok = max(t.shape[0] for t in batch_tokens)

    padded_cond = mx.zeros((len(batch_indices), max_cond, batch_cond[0].shape[-1]))
    padded_tokens = mx.zeros((len(batch_indices), max_tok), dtype=mx.int32)
    padded_targets = mx.full((len(batch_indices), max_tok), -1, dtype=mx.int32)  # -1 = ignore

    for i, (c, t, tgt) in enumerate(zip(batch_cond, batch_tokens, batch_targets)):
        padded_cond[i, :c.shape[0], :] = c
        padded_tokens[i, :t.shape[0]] = t
        padded_targets[i, :tgt.shape[0]] = tgt

    return padded_cond, padded_tokens, padded_targets


def loss_fn(model, cond, tokens, targets):
    """Cross-entropy loss for AR next-token prediction."""
    logits = model(tokens, cond_embeds=cond)  # (B, T, V)
    B, T, V = logits.shape

    # Flatten for cross entropy
    logits_flat = logits.reshape(-1, V)
    targets_flat = targets.reshape(-1)

    # Mask out padding (-1 targets)
    valid = targets_flat >= 0
    if not mx.any(valid):
        return mx.array(0.0)

    # Cross entropy on valid positions
    log_probs = logits_flat - mx.logsumexp(logits_flat, axis=-1, keepdims=True)
    target_log_probs = mx.take_along_axis(
        log_probs, mx.clip(targets_flat, 0, V - 1).reshape(-1, 1), axis=1
    ).squeeze(-1)

    loss = -mx.where(valid, target_log_probs, mx.zeros_like(target_log_probs))
    return loss.sum() / mx.maximum(valid.sum(), mx.array(1.0))


def evaluate(model, samples, batch_size=4):
    """Compute average loss and token-level accuracy."""
    total_loss = 0.0
    total_correct = 0
    total_tokens = 0

    indices = list(range(len(samples)))
    for start in range(0, len(indices), batch_size):
        batch_idx = indices[start:start + batch_size]
        cond, tokens, targets = make_training_batch(samples, batch_idx)

        logits = model(tokens, cond_embeds=cond)
        preds = logits.argmax(axis=-1)

        valid = targets >= 0
        correct = mx.sum((preds == targets) & valid)
        n_valid = mx.sum(valid)

        loss = loss_fn(model, cond, tokens, targets)
        mx.eval(loss, correct, n_valid)

        total_loss += loss.item() * n_valid.item()
        total_correct += correct.item()
        total_tokens += n_valid.item()

    avg_loss = total_loss / max(total_tokens, 1)
    accuracy = total_correct / max(total_tokens, 1)
    return avg_loss, accuracy


def main():
    parser = argparse.ArgumentParser(description="Train draft model locally (MLX)")
    parser.add_argument("--cache_dir", type=str, default="./cache_local")
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden_size", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--save_path", type=str, default="./checkpoints_local/draft.safetensors")
    args = parser.parse_args()

    # Load data
    samples = load_cached_dataset(args.cache_dir)
    if len(samples) < 2:
        print("Need at least 2 samples. Run cache_teacher.py first.")
        return

    # Split: 80% train, 20% eval
    split = max(1, int(len(samples) * 0.8))
    train_samples = samples[:split]
    eval_samples = samples[split:] if split < len(samples) else samples[:2]

    # Create model
    config = DraftMLXConfig(
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        num_heads=max(1, args.hidden_size // 64),
    )
    model = DraftModelMLX(config)
    mx.eval(model.parameters())

    num_params = sum(p.size for p in model.parameters().values() if isinstance(p, mx.array))
    print(f"Draft model: {num_params:,} parameters")
    print(f"Config: hidden={config.hidden_size}, layers={config.num_layers}, heads={config.num_heads}")

    # Optimizer
    optimizer = optim.AdamW(learning_rate=args.lr)
    loss_and_grad = nn.value_and_grad(model, loss_fn)

    # Training loop
    print(f"\nTraining for {args.num_epochs} epochs on {len(train_samples)} samples...")
    indices = list(range(len(train_samples)))

    for epoch in range(args.num_epochs):
        np.random.shuffle(indices)
        epoch_loss = 0.0
        num_batches = 0
        t0 = time.perf_counter()

        for start in range(0, len(indices), args.batch_size):
            batch_idx = indices[start:start + args.batch_size]
            cond, tokens, targets = make_training_batch(train_samples, batch_idx)

            loss, grads = loss_and_grad(model, cond, tokens, targets)
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state)

            epoch_loss += loss.item()
            num_batches += 1

        avg_loss = epoch_loss / max(num_batches, 1)
        elapsed = time.perf_counter() - t0

        # Eval
        eval_loss, eval_acc = evaluate(model, eval_samples, args.batch_size)

        print(f"  Epoch {epoch+1}/{args.num_epochs} | "
              f"train_loss={avg_loss:.4f} | eval_loss={eval_loss:.4f} | "
              f"eval_acc={eval_acc:.1%} | {elapsed:.1f}s")

    # Save checkpoint
    save_path = Path(args.save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    model.save_weights(str(save_path))
    print(f"\nSaved draft model to {save_path}")

    # Quick generation test
    print("\n--- Generation test ---")
    test_sample = samples[0]
    cond = mx.expand_dims(test_sample["cond_hidden"], axis=0)
    target_len = test_sample["cb0_tokens"].shape[0]

    generated = model.generate_ar(cond, num_tokens=target_len)
    mx.eval(generated)

    actual = test_sample["cb0_tokens"]
    match = mx.sum(generated == actual).item()
    print(f"Generated {target_len} tokens, {match}/{target_len} match teacher ({match/target_len:.1%})")
    print(f"First 20 generated: {np.array(generated[:20]).tolist()}")
    print(f"First 20 teacher:   {np.array(actual[:20]).tolist()}")


if __name__ == "__main__":
    main()
