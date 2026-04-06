"""PyTorch training script for RunPod A100.

Trains the AR draft model on cached teacher outputs using
cross-entropy + optional KL distillation loss.

Usage (on RunPod):
    python src/train.py \
        --cache_dir ./cache \
        --hidden_size 512 \
        --num_layers 6 \
        --batch_size 64 \
        --num_epochs 5 \
        --lr 3e-4
"""

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataset import TeacherCacheDataset, collate_fn
from draft_model import DraftModel


def train_epoch(model, dataloader, optimizer, scheduler, device, grad_clip=1.0, noise_sigma=0.0):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_tokens = 0

    for batch in dataloader:
        cond = batch["cond_hidden"].to(device)
        tokens = batch["input_tokens"].to(device)
        targets = batch["target_tokens"].to(device)

        # Noise augmentation: simulate framework numerical differences
        # Adds Gaussian noise to conditioning hidden states during training
        # so the draft learns to be robust to MLX vs PyTorch divergence
        if noise_sigma > 0:
            cond = cond + torch.randn_like(cond) * noise_sigma

        logits = model(tokens, cond_embeds=cond)  # (B, T, V)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1),
            ignore_index=-100,
        )

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        # Accuracy
        valid = targets != -100
        preds = logits.argmax(dim=-1)
        correct = ((preds == targets) & valid).sum().item()
        n_valid = valid.sum().item()

        total_loss += loss.item() * n_valid
        total_correct += correct
        total_tokens += n_valid

    return total_loss / max(total_tokens, 1), total_correct / max(total_tokens, 1)


@torch.no_grad()
def evaluate(model, dataloader, device):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_tokens = 0

    for batch in dataloader:
        cond = batch["cond_hidden"].to(device)
        tokens = batch["input_tokens"].to(device)
        targets = batch["target_tokens"].to(device)

        logits = model(tokens, cond_embeds=cond)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1),
            ignore_index=-100,
        )

        valid = targets != -100
        preds = logits.argmax(dim=-1)
        correct = ((preds == targets) & valid).sum().item()
        n_valid = valid.sum().item()

        total_loss += loss.item() * n_valid
        total_correct += correct
        total_tokens += n_valid

    return total_loss / max(total_tokens, 1), total_correct / max(total_tokens, 1)


def main():
    parser = argparse.ArgumentParser(description="Train draft model (PyTorch, A100)")
    parser.add_argument("--cache_dir", type=str, default="./cache")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints")
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=6)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--eval_split", type=float, default=0.1)
    parser.add_argument("--noise_sigma", type=float, default=0.0,
                        help="Gaussian noise sigma for conditioning augmentation (0.02-0.04 for MLX robustness)")
    parser.add_argument("--finetune_from", type=str, default="",
                        help="Path to checkpoint to fine-tune from (e.g. for MLX calibration)")
    parser.add_argument("--log_dir", type=str, default="./runs")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Load dataset
    full_dataset = TeacherCacheDataset(args.cache_dir)
    n_eval = max(1, int(len(full_dataset) * args.eval_split))
    n_train = len(full_dataset) - n_eval
    train_dataset, eval_dataset = torch.utils.data.random_split(
        full_dataset, [n_train, n_eval]
    )

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=4, pin_memory=True,
    )
    eval_loader = DataLoader(
        eval_dataset, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=2, pin_memory=True,
    )

    print(f"Dataset: {n_train} train, {n_eval} eval samples")

    # Create model
    model = DraftModel(
        vocab_size=1025,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        cond_dim=1024,
        dropout=0.1,
    ).to(device)

    # Load fine-tune checkpoint if provided
    if args.finetune_from:
        ckpt = torch.load(args.finetune_from, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"Fine-tuning from {args.finetune_from} (epoch {ckpt.get('epoch', '?')})")

    print(f"Draft model: {model.count_params():,} parameters")
    if args.noise_sigma > 0:
        print(f"Noise augmentation: sigma={args.noise_sigma}")

    # Optimizer + scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
    )

    total_steps = len(train_loader) * args.num_epochs
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr, total_steps=total_steps,
        pct_start=min(0.1, args.warmup_steps / max(total_steps, 1)),
    )

    writer = SummaryWriter(args.log_dir)
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Training loop
    best_eval_loss = float("inf")
    global_step = 0

    print(f"\nTraining for {args.num_epochs} epochs ({total_steps} steps)...")

    for epoch in range(args.num_epochs):
        t0 = time.perf_counter()

        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, scheduler, device, args.grad_clip,
            noise_sigma=args.noise_sigma,
        )
        eval_loss, eval_acc = evaluate(model, eval_loader, device)

        elapsed = time.perf_counter() - t0
        global_step += len(train_loader)

        writer.add_scalar("loss/train", train_loss, epoch)
        writer.add_scalar("loss/eval", eval_loss, epoch)
        writer.add_scalar("accuracy/train", train_acc, epoch)
        writer.add_scalar("accuracy/eval", eval_acc, epoch)
        writer.add_scalar("lr", optimizer.param_groups[0]["lr"], epoch)

        print(f"  Epoch {epoch+1}/{args.num_epochs} | "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.1%} | "
              f"eval_loss={eval_loss:.4f} eval_acc={eval_acc:.1%} | "
              f"{elapsed:.1f}s")

        # Save best
        if eval_loss < best_eval_loss:
            best_eval_loss = eval_loss
            torch.save({
                "model_state_dict": model.state_dict(),
                "config": {
                    "hidden_size": args.hidden_size,
                    "num_layers": args.num_layers,
                    "num_heads": args.num_heads,
                },
                "epoch": epoch,
                "eval_loss": eval_loss,
                "eval_acc": eval_acc,
            }, checkpoint_dir / "best.pt")
            print(f"    -> Saved best model (eval_loss={eval_loss:.4f})")

        # Save periodic
        torch.save({
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
        }, checkpoint_dir / "latest.pt")

    writer.close()
    print(f"\nTraining complete. Best eval loss: {best_eval_loss:.4f}")
    print(f"Checkpoints saved to {checkpoint_dir}/")


if __name__ == "__main__":
    main()
