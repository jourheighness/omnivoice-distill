"""Train token-conditioned draft model v2.

Usage:
    python src/train_v2.py \
        --cache_dir ./cache_v2 \
        --teacher_weights ./weights/omnivoice/model.safetensors \
        --hidden_size 512 --num_layers 6 --num_heads 8 \
        --batch_size 32 --num_epochs 20 --lr 3e-4
"""

import argparse
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataset_v2 import TeacherCacheDatasetV2, collate_v2
from draft_model_v2 import DraftModelV2


def train_epoch(model, dataloader, optimizer, scheduler, device, grad_clip=1.0):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_tokens = 0

    for batch in dataloader:
        cond_ids = batch["cond_ids"].to(device)
        audio_mask = batch["audio_mask"].to(device)
        tokens = batch["input_tokens"].to(device)
        targets = batch["target_tokens"].to(device)

        logits = model(tokens, cond_ids=cond_ids, audio_mask=audio_mask)
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
        cond_ids = batch["cond_ids"].to(device)
        audio_mask = batch["audio_mask"].to(device)
        tokens = batch["input_tokens"].to(device)
        targets = batch["target_tokens"].to(device)

        logits = model(tokens, cond_ids=cond_ids, audio_mask=audio_mask)
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_dir", required=True)
    parser.add_argument("--teacher_weights", required=True,
                        help="Path to teacher model.safetensors for embedding weights")
    parser.add_argument("--checkpoint_dir", default="./checkpoints_v2")
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=6)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--eval_split", type=float, default=0.1)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}")

    # Dataset
    full_dataset = TeacherCacheDatasetV2(args.cache_dir)
    n_eval = max(1, int(len(full_dataset) * args.eval_split))
    n_train = len(full_dataset) - n_eval
    train_dataset, eval_dataset = torch.utils.data.random_split(full_dataset, [n_train, n_eval])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate_v2, num_workers=4, pin_memory=True)
    eval_loader = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False,
                             collate_fn=collate_v2, num_workers=2, pin_memory=True)

    print(f"Dataset: {n_train} train, {n_eval} eval")

    # Model
    model = DraftModelV2(
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        dropout=0.1,
    ).to(device)

    # Load teacher embeddings (frozen)
    print("Loading teacher embeddings...")
    model.load_teacher_embeddings(args.teacher_weights, device=device)

    trainable = model.count_params(trainable_only=True)
    total = model.count_params(trainable_only=False)
    print(f"Draft model: {trainable:,} trainable / {total:,} total params")

    # Optimizer (only trainable params)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=args.weight_decay,
    )

    total_steps = len(train_loader) * args.num_epochs
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr, total_steps=total_steps,
        pct_start=0.1,
    )

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    best_eval_loss = float("inf")
    print(f"\nTraining for {args.num_epochs} epochs ({total_steps} steps)...")

    for epoch in range(args.num_epochs):
        t0 = time.perf_counter()
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, scheduler, device, args.grad_clip)
        eval_loss, eval_acc = evaluate(model, eval_loader, device)
        elapsed = time.perf_counter() - t0

        print(f"  Epoch {epoch+1}/{args.num_epochs} | "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.1%} | "
              f"eval_loss={eval_loss:.4f} eval_acc={eval_acc:.1%} | "
              f"{elapsed:.1f}s")

        if eval_loss < best_eval_loss:
            best_eval_loss = eval_loss
            torch.save({
                "model_state_dict": {k: v for k, v in model.state_dict().items()
                                     if "text_embed" not in k and "audio_embed" not in k},
                "config": {
                    "hidden_size": args.hidden_size,
                    "num_layers": args.num_layers,
                    "num_heads": args.num_heads,
                },
                "epoch": epoch,
                "eval_loss": eval_loss,
                "eval_acc": eval_acc,
            }, checkpoint_dir / "best.pt")
            print(f"    -> Saved best (eval_loss={eval_loss:.4f})")

    print(f"\nDone. Best eval loss: {best_eval_loss:.4f}")


if __name__ == "__main__":
    main()
